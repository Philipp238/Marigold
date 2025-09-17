#!/usr/bin/env bash
python eval.py \
    --base_data_dir /home/math/kneissl/MarigoldData/ \
    --dataset_config config/dataset/data_diode_all.yaml \
    --alignment least_square \
    --prediction_dir $prediction_path/diode/$checkpoint_name \
    --output_dir output/eval/diode/$model_name/median_ensemble_alignment \
    --aggregation median \
    --ensemble_alignment \
    --no_cuda
    