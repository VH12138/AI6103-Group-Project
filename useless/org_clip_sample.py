import gc
import io
import math
import sys
import os

import time
import datetime

import lpips
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def main():
# ====================================================== config setting =================================================== #
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',  # Modify this value to decrease the number of timesteps.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': False,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })

    prompts = ['A photo of a bedroom.']
    image_prompts = []
    batch_size = 1
    clip_guidance_scale = 1000  # Controls how much the image should look like the prompt.
    tv_scale = 150              # Controls the smoothness of the final output.
    range_scale = 50            # Controls how far out of range RGB values are allowed to be.
    cutn = 16
    n_batches = 10
    init_image = None   # This can be an URL or Colab local path and must be in quotes.
    skip_timesteps = 0  # This needs to be between approx. 200 and 500 when using an init image.
    init_scale = 0      # This enhances the effect of the init image, a good value is 1000.
    seed = 0

# ======================================================= Load models ============================================================ #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load('/home/jhan/6103/AI6103-Group-Project/models/lsun_bedroom.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()

    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    if seed is not None:
        torch.manual_seed(seed)

    make_cutouts = MakeCutouts(clip_size, cutn)
    side_x = side_y = model_config['image_size']

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = clip_model.encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([weight / cutn] * cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, out, y=None):
        n = x.shape[0]
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)
        clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
        image_embeds = clip_model.encode_image(clip_in).float()
        dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
        dists = dists.view([cutn, n, -1])
        losses = dists.mul(weights).sum(2).mean(0)
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out['pred_xstart'])
        loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
        if init is not None and init_scale:
            init_losses = lpips_model(x_in, init)
            loss = loss + init_losses.sum() * init_scale
        return -torch.autograd.grad(loss, x)[0]

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop

    all_images = []
    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        sample = sample_fn(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
            cond_fn_with_grad=True,
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.extend([sample.cpu().numpy()])
        print(f"created {len(all_images) * batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: n_batches]
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(
        '/home/jhan/6103/AI6103-Group-Project/output', 
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), 
        )
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_path = os.path.join(out_path, 'bedroom_{}x256x256.npz'.format(n_batches))
    print(f"saving to {file_path}")
    np.savez(file_path, arr)

    print("sampling complete")


if __name__ == "__main__":
    gc.collect()
    main()