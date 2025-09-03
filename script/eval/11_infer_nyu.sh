#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"output/25_08_20-17_06_23-train_marigold/checkpoint/iter_012000/unet/diffusion_pytorch_model.bin"}
mrgld_path=${2:-"/home/groups/ai/scholl/diffusion/models/marigold_depth-v1-0/"}
subfolder=${3:-"eval"}
BASE_DATA_DIR=${4:-"/scratch/scholl/diffusion/marigold_data/"}


python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --processing_res 0 \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --output_dir /scratch/scholl/diffusion/output/${subfolder}/nyu_test/prediction \
