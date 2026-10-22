"""
[file]          diagnose_density_input_dependence.py
[description]   Diagnose why a trained density-map checkpoint transfers poorly to a held-out room. The
                dense-BCE objective admits a degenerate solution where the decoder emits the room's
                marginal map and ignores the CSI input; this script measures that directly and also
                checks whether the empty gate can separate empty from occupied frames on this room.

                Three input conditions are scored on the same samples:
                  - real: the CSI as loaded,
                  - shuffled: the CSI rows permuted across samples, so each map is paired with another
                    sample's input; if the model reads the input, the maps change a lot,
                  - zeroed: all-zero CSI.
                Report the mean absolute map change relative to the real-map magnitude. Near-zero
                change on shuffled/zeroed means the model ignores the input (the shortcut), and no
                amount of target/decoder tuning will fix room transfer until that is addressed.

                Run with the same preprocessing config the checkpoint was trained with, e.g.
                AMAR_CSI_PREPROCESS=static,temporal,zscore python scripts/diagnose_density_input_dependence.py \
                    --checkpoint visualizations/cross_domain/meeting_room_classroom/5/model.pth --env empty_room
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
from src.models.density_map import (load_density_checkpoint, density_map_maxima, calibrate_empty_threshold)
from src.utils import select_device


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--checkpoint", required=True, type=str,
                          help="path to a density_map model.pth saved by save_density_checkpoint")
    var_args.add_argument("--env", required=True, type=str,
                          help="test room whose frames are scored, e.g. empty_room")
    var_args.add_argument("--backbone", default=None, type=str,
                          help="override nn.backbone (e.g. thatv-nostats). Default: the value stored "
                               "in the checkpoint, else inferred from its state-dict keys")
    var_args.add_argument("--encoder-norm", default=None, type=str,
                          help="override nn.encoder_norm (batch|group|layer). Default: the value stored "
                               "in the checkpoint, else inferred from its state-dict keys")
    var_args.add_argument("--users", default="0,1,2,3,4,5", type=str,
                          help="Comma-separated list of user IDs")
    var_args.add_argument("--batch-size", default=32, type=int)
    var_args.add_argument("--max-samples", default=None, type=int,
                          help="score at most this many frames (default: all)")
    var_args.add_argument("--out", default=None, type=str,
                          help="optional path of the diagnostic figure (PNG)")
    #
    return var_args.parse_args()


def load_model(var_checkpoint_path, var_device, var_backbone=None, var_encoder_norm=None):
    """
    [description]
    : rebuild the DensityMapNet described by the checkpoint and load its weights. The backbone and
      encoder norm are read from the checkpoint, else inferred from the state-dict keys, so a
      checkpoint trained with nn.backbone=thatv* or a non-batch nn.encoder_norm loads correctly.
    : return: (model, checkpoint dict)
    """
    #
    return load_density_checkpoint(var_checkpoint_path, var_device,
                                   var_backbone=var_backbone, var_encoder_norm=var_encoder_norm)


def predict_maps(var_model, var_x, var_device, var_batch_size):
    """
    [description]
    : forward the model over a whole array and return the sigmoid density maps (N, grid, grid).
    """
    #
    var_maps = []
    with torch.no_grad():
        for var_start in range(0, len(var_x), var_batch_size):
            var_batch = torch.from_numpy(var_x[var_start:var_start + var_batch_size]).to(var_device)
            var_density, _, _ = var_model(var_batch)
            var_maps.append(var_density.cpu().numpy())
    return np.concatenate(var_maps, axis=0) if var_maps else np.zeros((0, 1, 1), dtype=np.float32)


def _relative_change(var_maps, var_reference):
    """
    [description]
    : mean absolute difference to the reference maps, and the same difference normalized by the
      reference magnitude, so runs with different map scales are comparable.
    """
    #
    var_diff = float(np.mean(np.abs(var_maps - var_reference)))
    var_scale = float(np.mean(np.abs(var_reference)))
    return var_diff, (var_diff / var_scale if var_scale > 0 else float("nan"))


def run():
    var_args = parse_args()
    var_device = select_device()
    var_model, _ = load_model(var_args.checkpoint, var_device,
                              var_backbone=var_args.backbone,
                              var_encoder_norm=var_args.encoder_norm)
    print(f"Model: nn.backbone={preset['nn']['backbone']} nn.encoder_norm={preset['nn']['encoder_norm']}")
    #
    ## --------------------------------------------- data ------------------------------------------- ##
    #
    var_users = [var_user.strip() for var_user in var_args.users.split(",")]
    var_data_pd_y = load_data_y(preset["path"]["data_y"], var_environment=[var_args.env],
                                var_wifi_band=preset["data"]["wifi_band"], var_num_users=var_users)
    var_label_list = var_data_pd_y["label"].to_list()
    if var_args.max_samples is not None:
        ## slice the labels before loading so a quick diagnostic does not read the whole room; take an
        ## even spread rather than the first N so the rare empty frames are represented
        var_keep = np.unique(np.linspace(0, len(var_label_list) - 1,
                                         min(var_args.max_samples, len(var_label_list))).astype(int))
        var_data_pd_y = var_data_pd_y.iloc[var_keep]
        var_label_list = [var_label_list[var_idx] for var_idx in var_keep]
    var_x = load_data_x(preset["path"]["data_x"], var_label_list)
    var_x = np.ascontiguousarray(var_x.reshape(var_x.shape[0], var_x.shape[1], -1), dtype=np.float32)
    var_y = encode_occupancy_y(var_data_pd_y, var_args.env)
    var_true_count = var_y.sum(axis=1).round().astype(int)
    print(f"Room {var_args.env}: {len(var_x)} frames, feature shape {var_x.shape[1:]}, "
          f"preprocessing {preset['data']['preprocess']} (env AMAR_CSI_PREPROCESS="
          f"{os.environ.get('AMAR_CSI_PREPROCESS')})")
    #
    ## ------------------------------------------- maps --------------------------------------------- ##
    #
    var_maps_real = predict_maps(var_model, var_x, var_device, var_args.batch_size)
    var_perm = np.random.RandomState(0).permutation(len(var_x))
    var_maps_shuffled = predict_maps(var_model, var_x[var_perm], var_device, var_args.batch_size)
    var_maps_zeroed = predict_maps(var_model, np.zeros_like(var_x), var_device, var_args.batch_size)
    var_maps_mean = np.repeat(var_maps_real.mean(axis=0, keepdims=True), len(var_x), axis=0)
    #
    var_counts_real = var_maps_real.sum(axis=(1, 2))
    var_counts_shuffled = var_maps_shuffled.sum(axis=(1, 2))
    var_counts_zeroed = var_maps_zeroed.sum(axis=(1, 2))
    #
    ## ----------------------------------------- report --------------------------------------------- ##
    #
    print("\nINPUT DEPENDENCE (mean |map(condition) - map(real)|, and /|map(real)|):")
    for var_name, var_maps, var_counts in (("shuffled", var_maps_shuffled, var_counts_shuffled),
                                           ("zeroed", var_maps_zeroed, var_counts_zeroed),
                                           ("mean predicted map (room marginal)", var_maps_mean, None)):
        var_abs, var_rel = _relative_change(var_maps, var_maps_real)
        var_r = (float(np.corrcoef(var_counts_real, var_counts)[0, 1])
                 if var_counts is not None and np.std(var_counts) > 0 and np.std(var_counts_real) > 0
                 else float("nan"))
        print(f"  {var_name:36s} abs={var_abs:.5f}  rel={var_rel:.3f}  count_corr={var_r:.3f}")
    print("  rel near 0 => the map barely depends on the input (the marginal-map shortcut).")
    #
    ## -------------------------------------- empty gate -------------------------------------------- ##
    #
    var_map_max = density_map_maxima(var_maps_real)
    var_empty = var_true_count == 0
    var_threshold, var_score = calibrate_empty_threshold(var_map_max, var_true_count)
    print(f"\nEMPTY GATE: {int(var_empty.sum())} empty / {int((~var_empty).sum())} occupied frames")
    print(f"  map max empty    {var_map_max[var_empty].mean():.4f} +- {var_map_max[var_empty].std():.4f}"
          if var_empty.any() else "  map max empty    (none)")
    print(f"  map max occupied {var_map_max[~var_empty].mean():.4f} +- {var_map_max[~var_empty].std():.4f}"
          if (~var_empty).any() else "  map max occupied (none)")
    print(f"  best gate threshold={var_threshold:.4f}  balanced empty/occupied acc={var_score:.3f}"
          " (fitted on these frames, so an upper bound; 0.5 is chance, 1.0 is cleanly separable)")
    #
    ## --------------------------------------- figure ----------------------------------------------- ##
    #
    if var_args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        var_fig, var_axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ## a room with no empty (or no occupied) frame leaves a class empty; an empty hist() with
        ## density=True warns, and axvline(+-inf) from the degenerate gate draws nothing meaningful
        if var_empty.any():
            var_axes[0].hist(var_map_max[var_empty], bins=40, alpha=0.6, label="empty", density=True)
        if (~var_empty).any():
            var_axes[0].hist(var_map_max[~var_empty], bins=40, alpha=0.6, label="occupied", density=True)
        if np.isfinite(var_threshold):
            var_axes[0].axvline(var_threshold, color="k", ls="--", label=f"gate {var_threshold:.2f}")
        var_axes[0].set_title("map max: empty vs occupied"); var_axes[0].legend()
        var_axes[1].scatter(var_counts_real, var_counts_shuffled, s=6, alpha=0.4)
        var_axes[1].set_xlabel("count real"); var_axes[1].set_ylabel("count shuffled")
        var_axes[1].set_title("count under input shuffle")
        var_axes[2].imshow(var_maps_real.mean(axis=0), origin="lower")
        var_axes[2].set_title("mean predicted map"); var_axes[2].axis("off")
        var_fig.tight_layout()
        var_fig.savefig(var_args.out, dpi=120)
        print(f"\nFigure: {var_args.out}")


if __name__ == "__main__":
    run()
