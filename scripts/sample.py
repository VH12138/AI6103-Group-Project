"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import gc
import io
import math
import sys
import argparse
import os
import time
import datetime

import blobfile as bf

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
import torchvision
from torchvision import utils
import math
import clip


from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


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


def haversine_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def totalvar_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def main():
    time0 = time.time
    args = create_argparser().parse_args()

    logger.configure()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    if args.use_fp16:
        model.convert_to_fp16()


    clip_guidance_scale = 1000 
    totalvar_scale = 150              
    cutn = 16  
    skip_timesteps = 0  
    seed = 0
    init = None
    cur_t = None


    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
    with open('ref/bedroom_instructions.txt', 'r') as f:
        instructions = f.readlines()
    instructions = [tmp.replace('\n', '') for tmp in instructions]
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
    if seed is not None:
        torch.manual_seed(seed)

    make_cutouts = MakeCutouts(clip_size, cutn)

    target_embeds, weights = [], []

    for prompt in instructions:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()


    def cond_fn(x, t, out, y=None):
        n = x.shape[0]
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)
        clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
        image_embeds = clip_model.encode_image(clip_in).float()
        hav_loss = haversine_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
        hav_loss = hav_loss.view([cutn, n, -1])
        loss1 = hav_loss.mul(weights).sum(2).mean(0)
        loss2 = totalvar_loss(x_in)
        loss = loss1.sum() * clip_guidance_scale + loss2.sum() * totalvar_scale
        return -torch.autograd.grad(loss, x)[0]


    logger.log("sampling...")
    all_images = []
    sample_fn = diffusion.p_sample_loop

    while len(all_images) * args.batch_size < args.num_samples:
        cur_t = diffusion.num_timesteps - skip_timesteps - 1
        model_kwargs = {}
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=None,
            randomize_class=True,
            cond_fn_with_grad=True,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.extend([sample.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(
        '/home/jhan/6103/AI6103-Group-Project/output', 
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), 
        )
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_path = os.path.join(out_path, 'bedroom_{}x256x256.npz'.format(args.num_samples))
    print(f"saving to {file_path}")
    np.savez(file_path, arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        use_ddim=False,
        model_path="",
        rescale_timesteps=True,
        use_checkpoint=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()