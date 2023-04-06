# AI6103 Group Project

This is the codebase for NTU MSAI AI6103 Group Project based on [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for classifier.

# Download pre-trained models

The checkpoints for the main model are taken from the original codebase [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [xh-liu/SDG_code](https://github.com/xh-liu/SDG_code).

Here are the download links for involved model checkpoint:

The used class-unconditional diffusion model:
 * LSUN bedroom: [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)

The finetuned CLIP image encoders on noisy image as classifier guidance:
 * LSUN bedroom semantic guidance: [clipbedroom.pt](https://hkuhk-my.sharepoint.com/:u:/g/personal/xihuiliu_hku_hk/EfVpSVSjAhlEpsBCxSwkBnQByUvgNZqr38bxnG6bDHuOZQ?e=bOgCZT)


# Sampling from the models

To sample from these models, you can use the `org_clip_sample.py`, `noisy_clip_sample.py`, and `noisy_clip_sample_dist.py` scripts.
Here, we provide flags for sampling from all of these models.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 1000"
```

## Model flags
 * Original CLIP model is used as the classifier guidance for unconditional class diffusion model. The model flags are pre-defined in the script, no extra action is required.

 * Finetuned CLIP model on noisy images is used as the classifier guidance for unconditional class diffusion model. The model flags are defined as:

 For script `noisy_clip_sample.py`:
 ```
 MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --model_path models/lsun_bedroom.pt"
 ```

 For script `noisy_clip_sample_dist.py`:
 ```
 MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
```


## Run models
 * For script `org_clip_sample.py`:
 ```
 python org_clip_sample.py
 ```

 * For script `noisy_clip_sample.py`, run the script on a single GPU and only small sample size is allowed:
 ```
 GUIDANCE_FLAGS="--data_dir ref/ref_bedroom --text_weight 160 --image_weight 0 --text_instruction_file ref/bedroom_instructions.txt --clip_path models/clip_bedroom.pt"
 CUDA_VISIBLE_DEVICES=0 python -u scripts/noisy_clip_sample.py --exp_name bedroom_language_guidance --single_gpu $MODEL_FLAGS $SAMPLE_FLAGS $GUIDANCE_FLAGS
 ```

 * For script `noisy_clip_sample_dist.py`, run the GPU on distributed GPU and allows large sample size:
 ```
 python scripts/noisy_clip_sample_dist.py $MODEL_FLAGS --model_path models/lsun_bedroom.pt --clip_path models/clip_bedroom.pt $SAMPLE_FLAGS
 ```
