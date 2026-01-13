# Last modified: 2024-05-24
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from marigold import MarigoldPipeline
from marigold.diffusionUQ.unet import UNet_diffusion_mean, UNet_diffusion_mixednormal, UNet_diffusion_mvnormal, UNet_diffusion_normal, UNet_diffusion_iDDPM
from marigold.util.ensemble import ensemble_align
from src.util.seeding import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/25_08_20-17_06_23-train_marigold/checkpoint/iter_012000/unet/diffusion_pytorch_model.bin",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--marigold_path",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Marigold path or hub name.",
    )


    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    # Eval in place configuration
    parser.add_argument(
        "--eval_in_place",
        action="store_true",
        help="Evaluate in place and delete the infered right away",
    )
    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )
    parser.add_argument(
        "--eval_output_dir", type=str, default="", help="Output directory for evaluation results."
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    marigold_path = args.marigold_path
    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    eval_in_place = args.eval_in_place
    alignment = args.alignment
    alignment_max_res = args.alignment_max_res
    eval_output_dir = args.eval_output_dir
    pred_suffix = ".npy"


    assert not eval_in_place or (eval_output_dir != "")

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    if eval_in_place:
        check_directory(eval_output_dir)
        os.makedirs(eval_output_dir, exist_ok=True)
        logging.info(f"eval output dir = {eval_output_dir}")
    else:
        check_directory(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"output dir = {output_dir}")

    # -------------------- Config --------------------

    directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path))))
    config_path = os.path.join(directory, "config.yaml")
    cfg = OmegaConf.load(config_path)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )

    gt_dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    gt_dataloader = DataLoader(gt_dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None


    # ---------------- Eval in Place ----------------
    if eval_in_place:
        from tabulate import tabulate
        from src.util import metric
        from marigold.diffusionUQ import losses

        from src.util.alignment import (
            align_depth_least_square,
            depth2disparity,
            disparity2depth,
        )
        from src.util.metric import MetricTracker

        eval_metrics = [
            "abs_relative_difference",
            "squared_relative_difference",
            "rmse_linear",
            "rmse_log",
            "log10",
            "delta1_acc",
            "delta2_acc",
            "delta3_acc",
            "i_rmse",
            "silog_rmse",
        ]

        uq_eval_metrics = [
            "energy_score_mask",
            "gaussiannll_mask",
            "coverage_mask",
            "crps_mask",
        ]

        # -------------------- Eval metrics --------------------
        metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
        uq_metric_funcs = [getattr(losses, _met) for _met in uq_eval_metrics]

        metric_tracker_dict = {}
        per_sample_filename_dict = {}

        metric_tracker = MetricTracker(*[m.__name__ for m in (metric_funcs + uq_metric_funcs)])
        metric_tracker.reset()

        # -------------------- Per-sample metric file head --------------------
        per_sample_filename = os.path.join(eval_output_dir, "per_sample_metrics.csv")
        # write title
        with open(per_sample_filename, "w+") as f:
            f.write("filename,")
            f.write(",".join([m.__name__ for m in (uq_metric_funcs+metric_funcs)]))
            f.write("\n")


    
    pipe = MarigoldPipeline.from_pretrained(
        marigold_path, variant=variant, torch_dtype=dtype
    )

    unet_weights_dict = torch.load(checkpoint_path)


    distributional_method = cfg.loss.kwargs.distributional_method
    pipe.distributional_method = distributional_method
    if distributional_method == 'deterministic':
        pipe.unet.load_state_dict(unet_weights_dict)
    else:
        old_conv_out = pipe.unet.conv_out
        pipe.unet.conv_out = torch.nn.Identity()
        
        if distributional_method == 'normal':
            unet_diffusion = UNet_diffusion_normal(
                backbone=pipe.unet,
                conv_out=old_conv_out
            )
        if distributional_method == 'iDDPM':
            # training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            #     os.path.join(
            #         base_ckpt_dir,
            #         cfg.trainer.training_noise_scheduler.pretrained_path,
            #         "scheduler",
            #     )
            # )
            unet_diffusion = UNet_diffusion_iDDPM(
                backbone=pipe.unet,
                conv_out=old_conv_out,
                beta=pipe.scheduler.betas
            )
        elif distributional_method == 'mvnormal':
            unet_diffusion = UNet_diffusion_mvnormal(
                backbone=pipe.unet,
                conv_out=old_conv_out
            )
        elif distributional_method == 'mixednormal':
            unet_diffusion = UNet_diffusion_mixednormal(
                backbone=pipe.unet,
                conv_out=old_conv_out,
                n_components=cfg.loss.kwargs.n_components
            )
        elif distributional_method == 'deterministic_multi_output_head':
            unet_diffusion = UNet_diffusion_mean(
                backbone=pipe.unet,
                conv_out=old_conv_out,
                num_out_heads=cfg.loss.kwargs.num_out_heads
            )
        unet_diffusion.load_state_dict(unet_weights_dict)
        pipe.unet = unet_diffusion
    
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for (batch, data) in tqdm(
            zip(dataloader,gt_dataloader),
            desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            # Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)

            # Predict depth
            ensemble_preds = []
            for _ in range(ensemble_size):
                pipe_out = pipe(
                    input_image,
                    denoising_steps=denoise_steps,
                    ensemble_size=1,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=0,
                    color_map=None,
                    show_progress_bar=False,
                    resample_method=resample_method,
                    )

                depth_pred: np.ndarray = pipe_out.depth_np

                ensemble_preds.append(np.expand_dims(depth_pred, axis=0))  # [1, H, W]

            depth_preds = np.concatenate(ensemble_preds, axis=0)  # [N, H, W]

            if eval_in_place:
                # GT data
                depth_raw_ts = data["depth_raw_linear"].squeeze()
                valid_mask_ts = data["valid_mask_raw"].squeeze()
                rgb_name = data["rgb_relative_path"][0]

                depth_raw = depth_raw_ts.numpy()
                valid_mask = valid_mask_ts.numpy()

                depth_raw_ts = depth_raw_ts.to(device)
                valid_mask_ts = valid_mask_ts.to(device)
                rgb_basename = os.path.basename(rgb_name)
                pred_basename = get_pred_name(
                    rgb_basename, dataset.name_mode, suffix=pred_suffix
                )
                pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)


                depth_preds_ensemble_aligned = ensemble_align(depth_preds)
                depth_pred = np.median(depth_preds_ensemble_aligned, axis=0)[0]

                # Align with GT using least square
                if "least_square" == alignment:
                    depth_pred, scale, shift = align_depth_least_square(
                        gt_arr=depth_raw,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                elif "least_square_disparity" == alignment:
                    # convert GT depth -> GT disparity
                    gt_disparity, gt_non_neg_mask = depth2disparity(
                        depth=depth_raw, return_mask=True
                    )
                    # LS alignment in disparity space
                    pred_non_neg_mask = depth_pred > 0
                    valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                    disparity_pred, scale, shift = align_depth_least_square(
                        gt_arr=gt_disparity,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_nonnegative_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                    # convert to depth
                    disparity_pred = np.clip(
                        disparity_pred, a_min=1e-3, a_max=None
                    )  # avoid 0 disparity
                    depth_pred = disparity2depth(disparity_pred)
                    
                # Now shift each depth_pred_ensemble_aligned sample with the same scale and shift
                for i in range(depth_preds_ensemble_aligned.shape[0]):
                    depth_preds_ensemble_aligned[i] = depth_preds_ensemble_aligned[i] * scale + shift

                depth_preds = depth_preds_ensemble_aligned.squeeze(1) # (N, H, W)
                
                # Clip to dataset min max
                depth_preds = np.clip(
                    depth_preds, a_min=dataset.min_depth, a_max=dataset.max_depth
                )
                # clip to d > 0 for evaluation
                depth_preds = np.clip(depth_preds, a_min=1e-6, a_max=None)
                # Evaluate (using CUDA if available)
                depth_preds_ts = torch.from_numpy(depth_preds).to(device)

                sample_metric = []
                
                for met_func in uq_metric_funcs:
                    _metric_name = met_func.__name__
                    try:
                        _metric = met_func(depth_preds_ts, depth_raw_ts, valid_mask_ts).item()
                    except torch.OutOfMemoryError:
                        _metric = met_func(depth_preds_ts.cpu(), depth_raw_ts.cpu(), valid_mask_ts.cpu()).item()
                    sample_metric.append(_metric.__str__())
                    metric_tracker.update(_metric_name, _metric)

                # Aggregate
                depth_pred = np.median(depth_preds_ensemble_aligned, axis=0)[0]
                    

                # Align with GT using least square
                if "least_square" == alignment:
                    depth_pred, scale, shift = align_depth_least_square(
                        gt_arr=depth_raw,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                elif "least_square_disparity" == alignment:
                    # convert GT depth -> GT disparity
                    gt_disparity, gt_non_neg_mask = depth2disparity(
                        depth=depth_raw, return_mask=True
                    )
                    # LS alignment in disparity space
                    pred_non_neg_mask = depth_pred > 0
                    valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                    disparity_pred, scale, shift = align_depth_least_square(
                        gt_arr=gt_disparity,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_nonnegative_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                    # convert to depth
                    disparity_pred = np.clip(
                        disparity_pred, a_min=1e-3, a_max=None
                    )  # avoid 0 disparity
                    depth_pred = disparity2depth(disparity_pred)

                
                # Clip to dataset min max
                depth_pred = np.clip(
                    depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
                )
                # clip to d > 0 for evaluation
                depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
                # To device
                depth_pred_ts = torch.from_numpy(depth_pred).to(device)
                
                for met_func in metric_funcs:
                    _metric_name = met_func.__name__
                    _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                    sample_metric.append(_metric.__str__())
                    metric_tracker.update(_metric_name, _metric)

                # Save per-sample metric
                with open(per_sample_filename, "a+") as f:
                    f.write(pred_name + ",")
                    f.write(",".join(sample_metric))
                    f.write("\n")
                                
                                
                continue


            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            pred_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".npy"
            )
            save_to = os.path.join(scene_dir, pred_basename)
            if os.path.exists(save_to):
                logging.warning(f"Existing file: '{save_to}' will be overwritten")

            np.save(save_to, depth_preds)

    # -------------------- Save metrics to file --------------------
    if eval_in_place:
        eval_text = f"Evaluation metrics:\n\
        of in_place predictions\n\
        on dataset: {gt_dataset.disp_name}\n\
        with samples in: {gt_dataset.filename_ls_path}\n"

        eval_text += f"min_depth = {gt_dataset.min_depth}\n"
        eval_text += f"max_depth = {gt_dataset.max_depth}\n"


        eval_text += tabulate(
            [metric_tracker.result().keys(), metric_tracker.result().values()]
        )

        metrics_filename = "eval_metrics"
        if alignment:
            metrics_filename += f"-{alignment}"
        metrics_filename += ".txt"

        _save_to = os.path.join(eval_output_dir, metrics_filename)
        with open(_save_to, "w+") as f:
            f.write(eval_text)
            logging.info(f"Evaluation metrics saved to {_save_to}")
