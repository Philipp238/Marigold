#!/usr/bin/env bash

python eval.py \
    --base_data_dir /home/math/kneissl/MarigoldData/ \
    --dataset_config config/dataset/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir $prediction_path/eth3d/$model_name \
    --output_dir output/eval/eth3d/$model_name/median_ensemble_alignment \
    --aggregation median \
    --ensemble_alignment \
    --no_cuda