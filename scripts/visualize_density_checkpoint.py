"""
[file]          visualize_density_checkpoint.py
[description]   Re-render a density-map figure from a saved checkpoint, using the current peak
                algorithm, without retraining.

                A run saves model.pth (weights + x_shape + grid/sigma/decoder config + output_mode) but
                not the predicted maps, so to redraw the figure with a changed peak rule we reload the
                checkpoint, run inference on the held-out room, and call visualize_density_map with the
                peak floor set to the validation-calibrated count threshold (the "new peak algorithm").

                The count threshold is reproduced exactly as in run_density_map_cross_domain: 10% of
                each training room is held out with RandomState(39), the model is run on it, and
                resolve_count_decision is called on the concatenated readouts.

                Usage:
                    python scripts/visualize_density_checkpoint.py \
                        --checkpoint visualizations/cross_domain/meeting_room_classroom/3/model.pth \
                        --env empty_room --train_envs meeting_room,classroom \
                        --out visualizations/cross_domain/meeting_room_classroom/3/empty_room_peakviz
"""
#
##

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from configs.preset import preset
from src.data.load_data import load_data_x, load_data_y, encode_occupancy_y
from src.models.density_map import (
    DensityMapNet, build_kernels, peak_normalized_kernels, render_density_targets,
    sample_location_occupancy, density_map_maxima, resolve_count_decision, visualize_density_map,
)
from src.utils import select_device


def parse_args():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--checkpoint", required=True, type=str, help="path of a model.pth from a density run")
    var_args.add_argument("--env", required=True, type=str, help="the room to visualize (the held-out/test room)")
    var_args.add_argument("--train_envs", default=None, type=str,
                          help="comma-separated training rooms, used to calibrate the count threshold "
                               "exactly as the run did. Omit to use --peak_floor instead.")
    var_args.add_argument("--peak_floor", default=None, type=float,
                          help="override the peak floor (absolute map value a peak must clear). "
                               "Default: the validation-calibrated count threshold.")
    var_args.add_argument("--out", default=None, type=str, help="output directory (default: <checkpoint dir>/peakviz_<env>)")
    var_args.add_argument("--num_samples", default=6, type=int)
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str)
    return var_args.parse_args()


def load_model(var_checkpoint_path, var_device):
    var_checkpoint = torch.load(var_checkpoint_path, map_location=var_device)
    var_layout_name = var_checkpoint["layout_name"]
    var_layout = preset["layouts"].get(var_layout_name) or preset["layouts"][preset["data"]["environment"][0]]
    var_model = DensityMapNet(tuple(var_checkpoint["x_shape"]), var_layout,
                              grid_size=var_checkpoint["grid_size"],
                              sigma=var_checkpoint["sigma"],
                              hidden_dim=var_checkpoint.get("decoder_hidden"),
                              dropout=var_checkpoint.get("decoder_dropout"),
                              var_output_mode=var_checkpoint.get("output_mode", "probability")).to(var_device)
    var_model.load_state_dict(var_checkpoint["model_state_dict"])
    var_model.eval()
    return var_model, var_checkpoint


def load_room(var_env, var_users):
    var_data_pd_y = load_data_y(preset["path"]["data_y"], var_environment=[var_env],
                                var_wifi_band=preset["data"]["wifi_band"],
                                var_num_users=[u.strip() for u in var_users.split(",")])
    var_x = load_data_x(preset["path"]["data_x"], var_data_pd_y["label"].to_list())
    var_x = var_x.reshape(var_x.shape[0], var_x.shape[1], -1)
    var_y = encode_occupancy_y(var_data_pd_y, var_env)
    return var_x, var_y


def predict_density(var_model, var_x, var_device, var_batch_size=32):
    var_pred = []
    with torch.no_grad():
        for var_i in range(0, len(var_x), var_batch_size):
            var_density, _, _ = var_model(torch.from_numpy(var_x[var_i:var_i + var_batch_size]).to(var_device))
            var_pred.append(var_density.cpu())
    return torch.cat(var_pred, dim=0)


def calibrate_count_threshold(var_model, var_train_envs, var_grid_size, var_sigma, var_device, var_users):
    """
    Reproduce the run's validation split and threshold: 10% of each training room, RandomState(39).
    """
    var_occ_all, var_count_all, var_max_all = [], [], []
    for var_env in var_train_envs:
        var_x, var_y = load_room(var_env, var_users)
        var_perm = np.random.RandomState(39).permutation(len(var_x))
        var_num_valid = max(1, int(round(0.1 * len(var_x))))
        var_idx = np.sort(var_perm[:var_num_valid])
        var_kernels = torch.from_numpy(build_kernels(preset["layouts"][var_env], var_grid_size, var_sigma)[0])
        var_density = predict_density(var_model, var_x[var_idx], var_device)
        var_occ_all.append(sample_location_occupancy(var_density, var_kernels).numpy())
        var_count_all.append(var_y[var_idx].sum(axis=1).round())
        var_max_all.append(density_map_maxima(var_density))
    var_occ = np.concatenate(var_occ_all)
    var_count = np.concatenate(var_count_all)
    var_map_max = np.concatenate(var_max_all)
    var_empty_threshold, var_count_threshold = resolve_count_decision(var_occ, var_count, var_map_max)
    return var_empty_threshold, var_count_threshold


def run():
    var_args = parse_args()
    var_device = select_device()
    print(f"Using device: {var_device}")

    var_model, var_checkpoint = load_model(var_args.checkpoint, var_device)
    var_grid_size = var_checkpoint["grid_size"]
    var_sigma = var_checkpoint["sigma"]
    print(f"Checkpoint: layout '{var_checkpoint['layout_name']}', grid {var_grid_size}, "
          f"sigma {var_sigma}, output_mode '{var_checkpoint.get('output_mode', 'probability')}'")

    # ---- peak floor ----
    if var_args.peak_floor is not None:
        var_peak_floor = float(var_args.peak_floor)
        print(f"Peak floor: {var_peak_floor:.3f} (explicit)")
    elif var_args.train_envs:
        var_train_envs = [e.strip() for e in var_args.train_envs.split(",")]
        var_empty_threshold, var_peak_floor = calibrate_count_threshold(
            var_model, var_train_envs, var_grid_size, var_sigma, var_device, var_args.users)
        print(f"Calibrated on train rooms {var_train_envs}: count threshold {var_peak_floor:.3f} "
              f"(empty gate {var_empty_threshold:.3f})")
    else:
        var_peak_floor = 0.5
        print("Peak floor: 0.5 (default; pass --train_envs to calibrate, or --peak_floor to override)")

    # ---- held-out room: predictions and rendered ground truth ----
    var_x, var_y = load_room(var_args.env, var_args.users)
    var_density_pred = predict_density(var_model, var_x, var_device)
    var_kernels = torch.from_numpy(build_kernels(preset["layouts"][var_args.env], var_grid_size, var_sigma)[0])
    var_true_density = render_density_targets(torch.from_numpy(var_y), var_kernels).numpy()
    var_pred_density = var_density_pred.numpy()

    var_out = var_args.out or os.path.join(os.path.dirname(var_args.checkpoint), f"peakviz_{var_args.env}")
    os.makedirs(var_out, exist_ok=True)
    var_fig = visualize_density_map(var_true_density, var_pred_density, var_out,
                                    var_num_samples=var_args.num_samples,
                                    var_threshold_frac=preset["density"]["peak_threshold"],
                                    var_tag=var_args.env,
                                    var_layout=preset["layouts"][var_args.env],
                                    var_true_occupancy=var_y,
                                    var_peak_floor=var_peak_floor)
    print(f"\nPeak floor used: {var_peak_floor:.3f}")
    print(f"Figure: {var_fig}")


if __name__ == "__main__":
    run()
