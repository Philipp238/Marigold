#!/usr/bin/env bash
set -euo pipefail
set -x

# -------------------------------
# Arguments
# -------------------------------
CKPT="$1"             # Checkpoint path
MARIGOLD_PATH="$2"    # Path to Marigold model
SUBFOLDER="$3"        # e.g. "eval"
DATA_DIR="$4"         # Path to dataset root
OUTPUT_DIR="$5"       # Base output dir
DISTR_METHOD="$6"     # Distribution method (not used here, but passed)
ENSEMBLE_SIZE="$7"    # Number of ensemble samples
DIFFUSION_STEPS="$8"  # Number of denoising steps
IDENTIFIER="$9"       # Unique run identifier
EVAL_IN_PLACE="${10}" # Evaluate in place


if [[ $EVAL_IN_PLACE == "--eval_in_place" ]]; then
    eval_in_place_options="--eval_in_place "
    eval_in_place_options+="--alignment least_square "
    eval_in_place_options+="--eval_output_dir output/$SUBFOLDER/scannet/${IDENTIFIER}_${SUBFOLDER}_metric "
fi

# -------------------------------
# Run inference
# -------------------------------
python infer.py \
    --checkpoint "$CKPT" \
    --seed 1234 \
    --base_data_dir "$DATA_DIR" \
    --denoise_steps "$DIFFUSION_STEPS" \
    --ensemble_size "$ENSEMBLE_SIZE" \
    --processing_res 0 \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --output_dir "$OUTPUT_DIR/$SUBFOLDER/scannet/prediction_$IDENTIFIER" \
    --marigold_path "$MARIGOLD_PATH" \
    $eval_in_place_options
