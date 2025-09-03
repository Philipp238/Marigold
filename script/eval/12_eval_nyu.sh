#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir /scratch/scholl/diffusion/marigold_data/ \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --alignment least_square \
    --prediction_dir /scratch/scholl/diffusion/output/eval/nyu_test/prediction \
    --output_dir output/eval/nyu_test/eval_metric
