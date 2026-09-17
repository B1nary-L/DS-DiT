#!/bin/bash

pretrained_model="preset/models/stable-diffusion-3-medium/sd3_medium.safetensors"
hr_dir="preset/datasets/SECOND/train/HR"
lr_dir="preset/datasets/SECOND/train/LR_x8"
ref_dir="preset/datasets/SECOND/train/Ref"
results_dir="results"

# Training parameters
max_steps=85000
global_batch_size=16
learning_rate=5e-5
weight_decay=0.001

# Data parameters
image_size=512
num_workers=4

# Logging
log_every=100
ckpt_every=20000

# Run training
torchrun --nproc_per_node=4 \
    train_sr.py \
    --pretrained_model $pretrained_model \
    --hr_dir $hr_dir \
    --lr_dir $lr_dir \
    --ref_dir $ref_dir \
    --results_dir $results_dir \
    --max_steps $max_steps \
    --global_batch_size $global_batch_size \
    --lr $learning_rate \
    --weight_decay $weight_decay \
    --image_size $image_size \
    --num_workers $num_workers \
    --log_every $log_every \
    --ckpt_every $ckpt_every
