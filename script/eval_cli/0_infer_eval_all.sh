#!/usr/bin/env bash
set -euo pipefail
set -x

# -------------------------------
# Default values
# -------------------------------
output_storage_dir="../../output/"
skip_jobs=0
eval_in_place="False"
distr_method=""
iter=""
ckpt_name=""
diffusion_timesteps=""
ensemble_size=""

# -------------------------------
# Parse arguments
# -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -distr_method)          distr_method="$2"; shift 2 ;;
    -iter)                  iter="$2"; shift 2 ;;
    -ckpt_name)             ckpt_name="$2"; shift 2 ;;
    -diffusion_timesteps)   diffusion_timesteps="$2"; shift 2 ;;
    -ensemble_size)         ensemble_size="$2"; shift 2 ;;
    -output_storage_dir)    output_storage_dir="$2"; shift 2 ;;
    -eval_in_place)         eval_in_place="True"; shift 1 ;;
    -skip_jobs)             skip_jobs="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 -distr_method <method> -iter <iter> -ckpt_name <name> -diffusion_timesteps <steps> -ensemble_size <size> [-output_storage_dir <dir>] [-skip_jobs <[0-4]>]"
      echo "Defaults:"
      echo "  output_storage_dir = ../../output/"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use -h for help."
      exit 1
      ;;
  esac
done

# -------------------------------
# Required args check
# -------------------------------
if [[ -z "$distr_method" || -z "$iter" || -z "$ckpt_name" || -z "$diffusion_timesteps" || -z "$ensemble_size" ]]; then
  echo "Error: missing required arguments."
  exit 1
fi

# -------------------------------
# Setup paths
# -------------------------------
echo "Running with:"
echo "  distr_method       = $distr_method"
echo "  iter               = $iter"
echo "  ckpt_name          = $ckpt_name"
echo "  diffusion_timesteps= $diffusion_timesteps"
echo "  ensemble_size      = $ensemble_size"
echo "  output_storage_dir = $output_storage_dir"
echo "  skip_jobs          = $skip_jobs"
echo "  eval_in_place      = $eval_in_place"

identifier="${ckpt_name}_iter-${iter}_T-${diffusion_timesteps}_N-${ensemble_size}"
ckpt="/home/math/kneissl/Projects/Marigold/output/${ckpt_name}/checkpoint/iter_${iter}/unet/diffusion_pytorch_model.bin"
mrgld_path="/home/groups/ai/scholl/diffusion/models/marigold_depth-v1-0/"
subfolder="eval"
data_dir="/scratch/scholl/diffusion/marigold_data/"

# -------------------------------
# Optional flags (only set if True)
# -------------------------------

if [[ "$eval_in_place" == "True" ]]; then
  eval_in_place="--eval_in_place"
else
  eval_in_place="--eval_separately"
fi

# -------------------------------
# Helper function
# -------------------------------
run_stage() {
  local dataset="$1"   # e.g. nyu, kitti
  local stage="$2"     # infer | eval
  local idx="$3"       # numeric prefix (11, 12, …)

  bash "script/eval_cli/${idx}_${stage}_${dataset}.sh" \
    "$ckpt" "$mrgld_path" "$subfolder" "$data_dir" "$output_storage_dir" \
    "$distr_method" "$ensemble_size" "$diffusion_timesteps" "$identifier" "$eval_in_place"
}

# -------------------------------
# Dataset job mapping
# -------------------------------
declare -A DATASETS=(
  [0]="nyu"
  [1]="kitti"
  [2]="eth3d"
  [3]="scannet"
  [4]="diode"
)

for i in {0..4}; do
  if [ "$skip_jobs" -le "$i" ]; then
    dataset="${DATASETS[$i]}"
    run_stage "$dataset" "infer" "$(( (i+1)*10 + 1 ))"
    if [[ $eval_in_place != "--eval_in_place" ]]; then
      run_stage "$dataset" "eval"  "$(( (i+1)*10 + 2 ))"
    fi
  fi
done
