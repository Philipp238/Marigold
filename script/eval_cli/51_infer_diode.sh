#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value


python infer.py  \
    --checkpoint $1 \
    --seed 1234 \
    --base_data_dir $4 \
    --denoise_steps $8 \
    --ensemble_size $7 \
    --dataset_config config/dataset/data_diode_all.yaml \
    --output_dir $5/$3/diode/prediction_$9 \
    --processing_res 640 \
    --resample_method bilinear \
    --marigold_path $2 \
