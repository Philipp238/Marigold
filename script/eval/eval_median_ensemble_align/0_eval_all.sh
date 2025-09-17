#!/usr/bin/env bash

# export checkpoint_name="normal_T_50_N_10"
# export checkpoint_name="prediction_25_08_20-17_06_23-train_marigold_normal_iter-012000_T-50_N-10"
export checkpoint_name="prediction_25_08_27-10_54_49-train_marigold_iter-012000_T-50_N-10"

# export model_name="normal_T_50_N_10"
# export model_name="prediction_25_08_20-17_06_23-train_marigold_normal_iter-012000_T-50_N-10"
export model_name="mvnormal_iter_012000_T_50_N_10"

# export prediction_path="/scratch/scholl/diffusion/output/eval/"
export prediction_path="/scratch/kneissl/diffusion/output/eval/"

bash script/eval/eval_median_ensemble_align/12_eval_nyu.sh
bash script/eval/eval_median_ensemble_align/22_eval_kitti.sh
bash script/eval/eval_median_ensemble_align/32_eval_eth3d.sh
bash script/eval/eval_median_ensemble_align/42_eval_scannet.sh
bash script/eval/eval_median_ensemble_align/52_eval_diode.sh