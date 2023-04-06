"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import datetime

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
import torch.nn.functional as F

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

from clip_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from clip_diffusion.clip_guidance import CLIP_gd
from clip_diffusion.image_datasets import load_ref_data
from clip_diffusion.misc import set_random_seed
from clip_diffusion.guidance import image_loss, text_loss
from clip_diffusion.image_datasets import _list_image_files_recursively


def main():
    time0 = time.time
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    clip_model, preprocess = clip.load('ViT-B/16', device='cuda')
    
    with open('ref/bedroom_instructions.txt', 'r') as f:
        instructions = f.readlines()
    instructions = [tmp.replace('\n', '') for tmp in instructions]
    imgs = [None]

    clip_ft = CLIP_gd(args)
    clip_ft.load_state_dict(th.load(args.clip_path, map_location='cpu'))
    clip_ft.eval()
    clip_ft = clip_ft.cuda()

    def cond_fn_sdg(x, t, y, **kwargs):
        assert y is not None
        with th.no_grad():
            text_features = clip_model.encode_text(y)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            image_features = clip_ft.encode_image_list(x_in, t)
            loss_text = text_loss(image_features, text_features, args)
            total_guidance = loss_text * 160 

            return th.autograd.grad(total_guidance.sum(), x_in)[0]


    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        else:
            instruction = instructions[0]
            text = clip.tokenize([instruction]).to('cuda')
            model_kwargs['y'] = text
            model_kwargs = {k: v.to('cuda') for k, v in model_kwargs.items()}
        cond_fn = cond_fn_sdg
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        use_ddim=False,
        model_path="",
        clip_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()