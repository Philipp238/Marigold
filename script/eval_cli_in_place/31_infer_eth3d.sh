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
    --dataset_config config/dataset/data_eth3d.yaml \
    --output_dir $5/$3/eth3d/prediction_$9 \
    --processing_res 756 \
    --resample_method bilinear \
    --marigold_path $2 \
    --eval_in_place True \
    --alignment least_square \
    --eval_output_dir output/$3/eth3d/$9_$3_metric \
    --alignment_max_res 1024 \
    --evaluate_agg_and_ens_combinations ${10} \
    --eval_default_setting ${11} \