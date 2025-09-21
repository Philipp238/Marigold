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
    --dataset_config config/dataset/data_nyu_test.yaml \
    --output_dir $5/$3/nyu_test/prediction_$9 \
    --marigold_path $2 \
