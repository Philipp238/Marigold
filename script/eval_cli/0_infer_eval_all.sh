#!/usr/bin/env bash
set -e
set -x

# Default values
output_storage_dir="/home/math/kneissl/Projects/Marigold/output/"   # default if not provided
skip_jobs=0
distr_method=""
iter=""
ckpt_name=""
diffusion_timesteps=""
ensemble_size=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -distr_method)
      distr_method="$2"
      shift 2
      ;;
    -iter)
      iter="$2"
      shift 2
      ;;
    -ckpt_name)
      ckpt_name="$2"
      shift 2
      ;;
    -diffusion_timesteps)
      diffusion_timesteps="$2"
      shift 2
      ;;
    -ensemble_size)
      ensemble_size="$2"
      shift 2
      ;;
    -output_storage_dir)
      output_storage_dir="$2"
      shift 2
      ;;
    -skip_jobs)
      skip_jobs="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 -distr_method <method> -iter <iter> -ckpt_name <name> -diffusion_timesteps <steps> -ensemble_size <size> [-output_storage_dir <dir>] [-skip_jobs <[0-4]>]"
      echo "Defaults:"
      echo "  output_storage_dir = /foo/bar"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use -h for help."
      exit 1
      ;;
  esac
done

# Ensure required args are provided
if [[ -z "$distr_method" || -z "$iter" || -z "$ckpt_name" || -z "$diffusion_timesteps" || -z "$ensemble_size" ]]; then
  echo "Error: missing required arguments."
  echo "Usage: $0 -distr_method <method> -iter <iter> -ckpt_name <name> -diffusion_timesteps <steps> -ensemble_size <size> [-output_storage_dir <dir>]"
  exit 1
fi

# Example usage of variables
echo "Running with:"
echo "  distr_method       = $distr_method"
echo "  iter               = $iter"
echo "  ckpt_name          = $ckpt_name"
echo "  diffusion_timesteps= $diffusion_timesteps"
echo "  ensemble_size      = $ensemble_size"
echo "  output_storage_dir = $output_storage_dir"
echo "  skip_jobs          = $skip_jobs"

identifier="${ckpt_name}_iter-${iter}_T-${diffusion_timesteps}_N-${ensemble_size}"
ckpt="/home/math/kneissl/Projects/Marigold/output/${ckpt_name}/checkpoint/iter_${iter}/unet/diffusion_pytorch_model.bin"
mrgld_path="/home/groups/ai/scholl/diffusion/models/marigold_depth-v1-0/"
subfolder="eval"
BASE_DATA_DIR="/home/math/kneissl/MarigoldData"
exp_name=$distr_method

if [ "$skip_jobs" -le 0 ]; then
   bash script/eval_cli/11_infer_nyu.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   bash script/eval_cli/12_eval_nyu.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   rm -rf ${output_storage_dir}/${subfolder}/nyu_test/prediction_${identifier}
fi

if [ "$skip_jobs" -le 1 ]; then
   bash script/eval_cli/21_infer_kitti.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   bash script/eval_cli/22_eval_kitti.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   rm -rf ${output_storage_dir}/${subfolder}/kitti_eigen_test/prediction_${identifier}
fi

if [ "$skip_jobs" -le 2 ]; then
   bash script/eval_cli/31_infer_eth3d.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   bash script/eval_cli/32_eval_eth3d.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   rm -rf ${output_storage_dir}/${subfolder}/eth3d/prediction_${identifier}
fi

if [ "$skip_jobs" -le 3 ]; then
   bash script/eval_cli/41_infer_scannet.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   bash script/eval_cli/42_eval_scannet.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   rm -rf ${output_storage_dir}/${subfolder}/scannet/prediction_${identifier}
fi

if [ "$skip_jobs" -le 4 ]; then
   bash script/eval_cli/51_infer_diode.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   bash script/eval_cli/52_eval_diode.sh $ckpt $mrgld_path $subfolder $BASE_DATA_DIR $output_storage_dir $exp_name $ensemble_size $diffusion_timesteps $identifier
   rm -rf ${output_storage_dir}/${subfolder}/diode/prediction_${identifier}
fi
