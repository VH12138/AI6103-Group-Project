# AI6103 Group Project

This is the codebase for NTU MSAI AI6103 Group Project based on [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for classifier.

# Download pre-trained models

The checkpoints for the main model are taken from the original codebase [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

Here are the download links for involved model checkpoint:

 * LSUN bedroom: [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)
 * LSUN cat: [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_cat.pt)
 * LSUN horse: [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_horse.pt)


# Sampling from the models

To sample from these models, you can use the `sample.py` script.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 1000"
```

## Model flags
 * CLIP model is used as the classifier guidance for unconditional class diffusion model.

 You can set the model flags as following:
 ```
 MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True" 
 ```


## Run models
 * Here we show how to sample from `lsun_bedroom.pt`, but the other two LSUN checkpoints should work as well:
 ```
 python scripts/sample.py $MODEL_FLAGS --model_path models/lsun_bedroom.pt --ref_path ref/bedroom_instruction.txt $SAMPLE_FLAGS
 ```
