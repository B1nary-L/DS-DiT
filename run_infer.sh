#!/bin/bash

image_size=512
steps=40
seed=0

autog=0                # set 1 to enable auto-guidance, 0 to disable
guidance=1.2           # w: 1.0-1.2
ref_scale_weak=0.0     # weak reference guidance scale; increasing this value allows a larger guidance scale


pretrained_model="preset/models/stable-diffusion-3-medium/sd3_medium.safetensors" # vae
checkpoint="preset/checkpoints/0080000.pt"
lr_dir="preset/datasets/SECOND/test/LR_x8"
ref_dir="preset/datasets/SECOND/test/Ref"
output_dir="preset/output"

autog_args=""
if [ "$autog" = 1 ]; then
    autog_args="--autog --guidance $guidance --ref_scale_weak $ref_scale_weak"
    output_dir="${output_dir}_w${guidance}"
fi

torchrun --nproc_per_node=4 infer_sr.py \
    --pretrained_model "$pretrained_model" \
    --checkpoint "$checkpoint" \
    --lr_dir "$lr_dir" \
    --ref_dir "$ref_dir" \
    --output_dir "$output_dir" \
    --image_size $image_size \
    --steps $steps \
    --seed $seed \
    $autog_args \
    --use_ddp
