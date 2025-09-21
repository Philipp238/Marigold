#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $4 \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --alignment least_square \
    --prediction_dir $5/$3/scannet/prediction_$9 \
    --output_dir output/$3/scannet/$9_$3_metric \
