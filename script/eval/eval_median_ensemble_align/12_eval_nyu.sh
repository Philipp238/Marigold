#!/usr/bin/env bash
# set -e
# set -x


python eval.py \
    --base_data_dir /home/math/kneissl/MarigoldData/ \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --alignment least_square \
    --prediction_dir $prediction_path/nyu_test/$model_name \
    --output_dir output/eval/nyu_test/$model_name/median_ensemble_alignment \
    --aggregation median \
    --ensemble_alignment \
    --no_cuda