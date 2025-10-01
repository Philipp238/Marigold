#!/usr/bin/env bash
set -euo pipefail
set -x

# -------------------------------
# Arguments
# -------------------------------
CKPT="$1"             # Checkpoint path (not used here)
MARIGOLD_PATH="$2"    # Path to Marigold model (not used here)
SUBFOLDER="$3"        # e.g. "eval"
DATA_DIR="$4"         # Path to dataset root
OUTPUT_DIR="$5"       # Base output dir
DISTR_METHOD="$6"     # Distribution method (not used here)
ENSEMBLE_SIZE="$7"    # Ensemble size (not used here)
DIFFUSION_STEPS="$8"  # Diffusion steps (not used here)
IDENTIFIER="$9"       # Unique run identifier

# -------------------------------
# Run evaluation
# -------------------------------
python eval.py \
    --base_data_dir "$DATA_DIR" \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --alignment least_square \
    --prediction_dir "$OUTPUT_DIR/$SUBFOLDER/nyu_test/prediction_$IDENTIFIER" \
    --output_dir "output/$SUBFOLDER/nyu_test/${IDENTIFIER}_${SUBFOLDER}_metric"
