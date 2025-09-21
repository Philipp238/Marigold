#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $4 \
    --dataset_config config/dataset/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir $5/$3/eth3d/prediction_$9 \
    --output_dir output/$3/eth3d/$9_$3_metric \
    --alignment_max_res 1024 \
