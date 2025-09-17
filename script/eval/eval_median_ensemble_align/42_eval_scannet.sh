#!/usr/bin/env bash

python eval.py \
    --base_data_dir /home/math/kneissl/MarigoldData/ \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --alignment least_square \
    --prediction_dir $prediction_path/scannet/$model_name \
    --output_dir output/eval/scannet/$model_name/median_ensemble_alignment \
    --aggregation median \
    --ensemble_alignment \
    --no_cuda