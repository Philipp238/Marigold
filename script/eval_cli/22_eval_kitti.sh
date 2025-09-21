#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $4 \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir $5/$3/kitti_eigen_test/prediction_$9 \
    --output_dir output/$3/kitti_eigen_test/$9_$3_metric
