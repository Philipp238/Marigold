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
    --processing_res 0 \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --output_dir $5/$3/scannet/prediction_$9 \
    --marigold_path $2 \
    --eval_in_place True \
    --alignment least_square \
    --eval_output_dir output/$3/scannet/$9_$3_metric \
