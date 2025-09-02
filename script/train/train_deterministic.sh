#!/usr/bin/env bash

python train.py \
    --config config/train_marigold_deterministic.yaml \
    --output_dir /scratch/scholl/diffusion/outputs/train/marigold_deterministic \
    --base_data_dir /scratch/scholl/diffusion/marigold_data \
    --base_ckpt_dir /home/groups/ai/scholl/diffusion/models/
