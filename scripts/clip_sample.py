"""
Generate a large batch of image samples from a class-unconditional diffusion model 
with CLIP as classifier guidance and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time 

import math 
import clip 
from torchvision import utils
from torchvision import transforms

import blobfile as bf 
import numpy as np
import torch as th
import torch.distributed as dist

import torchvision 
import torch.nn.functional as F

from guided_diffusion.image_datasets import load_ref_data
from guided_diffusion.guidance import text_loss 
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def cond_fn(x, t, y, **kwargs):
    assert y is not None
    with th.no_grad():
        if args.text_weight != 0:
            text_features = clip_model.encode_text(y)
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        clip_in = normalize(x_in)
        image_features = clip_model.encode_image(clip_in).float()
        if args.text_weight != 0:
            loss_text = text_loss(image_features, text_features, args)
        total_guidance = loss_text * args.text_weight

        return th.autograd.grad(total_guidance.sum(), x_in)[0]

def main():
    time0 = time.time()
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


# ======================================================= Change to CLIP ================================================= #
    clip_model, preprocess = clip.load('ViT-B/16', device='cuda')
    if args.text_weight == 0:
        instructions = [""]
    else:
        with open(args.text_instruction_file, 'r') as f:
            instructions = f.readlines()
    instructions = [tmp.replace('\n', '') for tmp in instructions]
    imgs = [None]
    clip_model.eval()

    logger.log("sampling...")
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
    count = 0
    for img_cnt in range(len(imgs)):
        if imgs[img_cnt] is None:
            model_kwargs = {}
        
        for ins_cnt in range(len(instructions)):
            instruction = instructions[ins_cnt]    
            text = clip.tokenize([instruction for cnt in range(args.batch_size)]).to('cuda')
            model_kwargs['y'] = text
            model_kwargs = {k: v.to('cuda') for k, v in model_kwargs.items()}
            
            with th.cuda.amp.autocast(True):
                sample = diffusion.p_sample_loop(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    noise=None,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device='cuda',
                )

            for i in range(args.batch_size):
                out_folder = '%05d_%s' % (ins_cnt, instructions[ins_cnt])
                out_path = os.path.join(args.logdir, out_folder, 
                                        f"{str(count * args.batch_size + i).zfill(5)}.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                utils.save_image(
                    sample[i].unsqueeze(0),
                    out_path,
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            
            count += 1
            logger.log(f"created {count * args.batch_size} samples")
            logger.log(time.time() - time0)
    
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        text_weight=160,
        text_instruction_file='ref/bedroom_instructions.txt',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
            

if __name__ == "__main__":
    main()
