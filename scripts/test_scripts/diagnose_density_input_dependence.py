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

                The empty-gate section then checks stage 1 of the count decision on this room. It
                reports, per per-frame map statistic (map max, map mean, max/mean, mass), the empty and
                occupied distributions and two thresholds: the shipped supervised calibration (an
                in-sample upper bound) and an unsupervised Otsu cut (what a room could self-calibrate
                to with no labels). Pass --source-empty-threshold with the EmptyThr the training run
                used to see whether that source-calibrated gate fires on this room at all - if it
                never does, every empty frame is counted as occupied. The scale-invariant max/mean is
                the candidate to survive the raised map floor that a domain shift produces.

                Run with the same preprocessing config the checkpoint was trained with, e.g.
                AMAR_CSI_PREPROCESS=static,temporal,zscore python scripts/diagnose_density_input_dependence.py \
                    --checkpoint visualizations/cross_domain/meeting_room_classroom/5/model.pth --env empty_room
"""
#
##
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from configs.preset import preset
from src.data.load_data import load_data_x, load_data_y, encode_occupancy_y
from src.models.density_map import density_map_maxima, calibrate_empty_threshold
from src.models.checkpoints import load_cross_domain_model, checkpoint_model_name
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
    var_args.add_argument("--source-empty-threshold", default=None, type=float,
                          help="the empty-gate threshold the training run actually used (calibrated on "
                               "the source rooms' validation split). Reports how often it fires on this "
                               "room, i.e. whether the gate transfers across the domain gap")
    var_args.add_argument("--out", default=None, type=str,
                          help="optional path of the diagnostic figure (PNG)")
    #
    return var_args.parse_args()


def load_model(var_checkpoint_path, var_device, var_backbone=None, var_encoder_norm=None):
    """
    [description]
    : rebuild any cross-domain model from its checkpoint -- density head, coords head, AMAR_WO_RVQ or
      the dummy baseline -- and return it with the dispatcher name of the architecture it is.
    : return: (model, checkpoint dict, model_name)
    """
    #
    var_model, var_checkpoint = load_cross_domain_model(var_checkpoint_path, var_device)
    var_name = checkpoint_model_name(var_checkpoint)
    ## For the density head an explicit backbone / encoder-norm override still wins (the stored values
    ## are already applied by load_density_checkpoint); for the other architectures the ctor recorded
    ## in the checkpoint fixes them, so an override would not rebuild a matching module.
    if var_name in ("density_map", "density_map_dem") and (var_backbone or var_encoder_norm):
        from src.models.density_map import load_density_checkpoint
        var_model, var_checkpoint = load_density_checkpoint(
            var_checkpoint_path, var_device, var_backbone=var_backbone, var_encoder_norm=var_encoder_norm)
    return var_model, var_checkpoint, var_name


def predict_outputs(var_model_name, var_model, var_x, var_device, var_batch_size):
    """
    [description]
    : forward any cross-domain model over a whole array.
    : return (flat (N, D), gate (N,) or None). flat is the per-frame output the relative-change
      comparison is computed on; gate is the per-frame statistic the model's own empty decision
      thresholds -- the density map maximum, or the maximum objectness probability -- or None when the
      architecture ships no empty gate (AMAR_WO_RVQ, the dummy baseline).
    """
    #
    var_flat, var_gate = [], []
    with torch.no_grad():
        for var_start in range(0, len(var_x), var_batch_size):
            var_batch = torch.from_numpy(var_x[var_start:var_start + var_batch_size]).to(var_device)
            if var_model_name in ("density_map", "density_map_dem"):
                var_density = var_model(var_batch)[0]
                var_flat.append(var_density.reshape(len(var_batch), -1).cpu().numpy())
                var_gate.append(var_density.amax(dim=(1, 2)).cpu().numpy())
            elif var_model_name == "density_map_coords":
                var_xy, var_obj = var_model(var_batch)
                var_prob = torch.sigmoid(var_obj)
                var_flat.append(torch.cat([var_xy, var_prob.unsqueeze(-1)], dim=-1)
                                .reshape(len(var_batch), -1).cpu().numpy())
                var_gate.append(var_prob.amax(dim=1).cpu().numpy())
            elif var_model_name == "AMAR_WO_RVQ":
                ## the decoder stacks its layers; the last one is the trained output (B, queries, classes)
                var_out = var_model(var_batch)[-1]
                var_flat.append(var_out.reshape(len(var_batch), -1).cpu().numpy())
            else:   ## dummy_model
                var_flat.append(var_model(var_batch).reshape(len(var_batch), -1).cpu().numpy())
    var_flat = np.concatenate(var_flat, axis=0) if var_flat else np.zeros((0, 1), dtype=np.float32)
    var_gate = np.concatenate(var_gate, axis=0) if var_gate else None
    return var_flat, var_gate


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


def _balanced_empty_accuracy(var_values, var_threshold, var_is_empty):
    """
    [description]
    : balanced accuracy of the decision "empty when the statistic is below the threshold", the same
      objective calibrate_empty_threshold optimizes. var_is_empty carries the ground truth and is used
      only to score; the threshold itself is chosen without labels by otsu_threshold.
    """
    if not var_is_empty.any() or var_is_empty.all():
        return float("nan")
    var_pred_empty = np.asarray(var_values) < var_threshold
    return float(0.5 * (np.mean(var_pred_empty[var_is_empty]) + np.mean(~var_pred_empty[~var_is_empty])))


def otsu_threshold(var_values, var_is_empty, var_num_bins=256):
    """
    [description]
    : Otsu's threshold on a per-frame map statistic: the cut that maximizes the between-class variance
      of the values alone. No labels choose it, so it is the gate a room could calibrate on its own
      unannotated frames; var_is_empty is used only to score the result.
    : return: (threshold, balanced_accuracy)
    """
    var_values = np.asarray(var_values, dtype=np.float64)
    var_lo, var_hi = float(var_values.min()), float(var_values.max())
    if var_hi <= var_lo:
        return var_lo, float("nan")
    var_hist, var_edges = np.histogram(var_values, bins=var_num_bins, range=(var_lo, var_hi))
    var_centers = (var_edges[:-1] + var_edges[1:]) / 2.0
    var_w0 = np.cumsum(var_hist).astype(np.float64)
    var_w1 = float(var_hist.sum()) - var_w0
    var_sum0 = np.cumsum(var_hist * var_centers)
    var_sum1 = var_sum0[-1] - var_sum0
    ## bins with an empty side cannot separate, so their between-class variance is undefined (nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_between = var_w0 * var_w1 * (var_sum0 / var_w0 - var_sum1 / var_w1) ** 2
    var_threshold = float(var_centers[int(np.argmax(np.nan_to_num(var_between)))])
    return var_threshold, _balanced_empty_accuracy(var_values, var_threshold, var_is_empty)


def run():
    var_args = parse_args()
    var_device = select_device()
    var_model, var_checkpoint, var_model_name = load_model(var_args.checkpoint, var_device,
                                                           var_backbone=var_args.backbone,
                                                           var_encoder_norm=var_args.encoder_norm)
    var_is_density = var_model_name in ("density_map", "density_map_dem")
    print(f"Model: {var_model_name} nn.backbone={preset['nn']['backbone']} "
          f"nn.encoder_norm={preset['nn']['encoder_norm']} "
          f"train_envs={var_checkpoint.get('train_envs')}")
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
    ## ------------------------------------------- outputs ------------------------------------------ ##
    #
    var_out_real, var_gate_real = predict_outputs(var_model_name, var_model, var_x, var_device,
                                                  var_args.batch_size)
    var_perm = np.random.RandomState(0).permutation(len(var_x))
    var_out_shuffled, _ = predict_outputs(var_model_name, var_model, var_x[var_perm], var_device,
                                          var_args.batch_size)
    var_out_zeroed, _ = predict_outputs(var_model_name, var_model, np.zeros_like(var_x), var_device,
                                        var_args.batch_size)
    var_out_mean = np.repeat(var_out_real.mean(axis=0, keepdims=True), len(var_x), axis=0)
    ## the per-frame response magnitude, correlated across conditions the way the density "count" was
    var_mag_real = var_out_real.sum(axis=1)
    var_mag_shuffled = var_out_shuffled.sum(axis=1)
    var_mag_zeroed = var_out_zeroed.sum(axis=1)
    #
    ## ----------------------------------------- report --------------------------------------------- ##
    #
    print("\nINPUT DEPENDENCE (mean |output(condition) - output(real)|, and /|output(real)|):")
    for var_name, var_out, var_mag in (("shuffled", var_out_shuffled, var_mag_shuffled),
                                       ("zeroed", var_out_zeroed, var_mag_zeroed),
                                       ("mean predicted output (room marginal)", var_out_mean, None)):
        var_abs, var_rel = _relative_change(var_out, var_out_real)
        ## corrcoef divides by the std; a near-constant series (e.g. all-zero input) makes that 0/0,
        ## so it returns nan and prints a RuntimeWarning. Detect the degenerate case explicitly and
        ## suppress the warning rather than let it litter the log.
        var_r = float("nan")
        if (var_mag is not None and np.std(var_mag) > 1e-12
                and np.std(var_mag_real) > 1e-12):
            with np.errstate(invalid="ignore", divide="ignore"):
                var_r = float(np.corrcoef(var_mag_real, var_mag)[0, 1])
        print(f"  {var_name:36s} abs={var_abs:.5f}  rel={var_rel:.3f}  resp_corr={var_r:.3f}")
    print("  rel near 0 => the output barely depends on the input (the marginal / prior shortcut).")
    #
    ## -------------------------------------- empty gate -------------------------------------------- ##
    #
    if var_gate_real is None:
        print(f"\nEMPTY GATE: n/a for {var_model_name} (it ships no empty-gate statistic).")
    else:
        var_empty = var_true_count == 0
        ## The gate is a 1-D threshold on a per-frame statistic of the real output; the density head
        ## thresholds the map maximum. A raised map floor (the cross-domain failure) lifts the maximum
        ## without making the map peaky, so for the density head the scale-invariant statistics are
        ## reported next to it: "mass" (= the count) and "map mean" both move when the whole map is
        ## lifted, while "max/mean" (peakiness) should not. The coords head thresholds its maximum
        ## objectness probability.
        if var_is_density:
            var_grid = int(var_checkpoint.get("grid_size", preset["density"]["grid_size"]))
            var_maps_real = var_out_real.reshape(len(var_x), var_grid, var_grid)
            var_gate_stat = density_map_maxima(var_maps_real)
            var_stats = {
                "map max": var_gate_stat,
                "map mean": var_maps_real.mean(axis=(1, 2)),
                "max/mean": var_gate_stat / (var_maps_real.mean(axis=(1, 2)) + 1e-8),
                "mass": var_maps_real.sum(axis=(1, 2)),
            }
            var_stat_label = "map max"
        else:
            var_gate_stat = var_gate_real
            var_stats = {"obj max": var_gate_stat}
            var_stat_label = "obj max"
        print(f"\nEMPTY GATE: {int(var_empty.sum())} empty / {int((~var_empty).sum())} occupied frames")
        print(f"  {'statistic':<11}{'empty':>17}{'occupied':>17}{'Otsu thr':>10}{'Otsu bal':>10}"
              f"{'sup thr':>10}{'sup bal':>9}")
        var_gate_threshold = None
        for var_name, var_values in var_stats.items():
            var_empty_str = (f"{var_values[var_empty].mean():.4f}+-{var_values[var_empty].std():.4f}"
                             if var_empty.any() else "(none)")
            var_occ_str = (f"{var_values[~var_empty].mean():.4f}+-{var_values[~var_empty].std():.4f}"
                           if (~var_empty).any() else "(none)")
            ## Otsu is label-free (a room could self-calibrate); the supervised one is the shipped rule
            ## and is fitted on these frames, so it is an in-sample upper bound
            var_otsu_thr, var_otsu_bal = otsu_threshold(var_values, var_empty)
            var_sup_thr, var_sup_bal = calibrate_empty_threshold(var_values, var_true_count)
            if var_name == var_stat_label:
                var_gate_threshold = var_sup_thr
            print(f"  {var_name:<11}{var_empty_str:>17}{var_occ_str:>17}"
                  f"{var_otsu_thr:>10.4f}{var_otsu_bal:>10.3f}{var_sup_thr:>10.4f}{var_sup_bal:>9.3f}")
        print("  gate fires when the statistic is BELOW the threshold. Otsu = label-free (self-calibrated "
              "per room); sup = fitted on these frames (upper bound). 0.5 = chance, 1.0 = clean.")
        #
        ## The threshold the training run actually used was calibrated on the source rooms; does it fire
        ## on this room? If it never fires, every empty frame is counted as occupied (count-0 accuracy 0).
        if var_args.source_empty_threshold is not None:
            var_src = float(var_args.source_empty_threshold)
            var_src_empty = var_gate_stat < var_src
            print(f"\n  SOURCE GATE (calibrated on the training rooms) = {var_src:.4f}")
            print(f"    fires on {int(var_src_empty[var_empty].sum())}/{int(var_empty.sum())} empty frames, "
                  f"{int(var_src_empty[~var_empty].sum())}/{int((~var_empty).sum())} occupied frames")
            if var_empty.any():
                print(f"    empty recall (= the count-0 accuracy it can reach) = "
                      f"{float(var_src_empty[var_empty].mean()):.3f}")
            if not var_src_empty.any():
                print("    NEVER FIRES on this room: the source threshold sits below this room's whole "
                      "range, so every empty frame is counted as occupied.")
            elif var_empty.any() and float(var_src_empty[var_empty].mean()) < 0.5:
                print("    Mostly dead on this room: the source threshold misses most empty frames.")
    #
    ## --------------------------------------- figure ----------------------------------------------- ##
    #
    if var_args.out and not var_is_density:
        print(f"\n(no diagnostic figure for {var_model_name}: the figure plots density-map statistics)")
    elif var_args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        var_fig, var_axes = plt.subplots(1, 4, figsize=(20, 4.5))

        def _panel(var_ax, var_values, var_title, var_thresholds):
            ## a room with no empty (or no occupied) frame leaves a class empty; an empty hist() with
            ## density=True warns, so each class is drawn only when present
            if var_empty.any():
                var_ax.hist(var_values[var_empty], bins=40, alpha=0.6, label="empty", density=True)
            if (~var_empty).any():
                var_ax.hist(var_values[~var_empty], bins=40, alpha=0.6, label="occupied", density=True)
            for var_label, var_value, var_style in var_thresholds:
                if var_value is not None and np.isfinite(var_value):
                    var_ax.axvline(var_value, color="k", ls=var_style, label=var_label)
            var_ax.set_title(var_title); var_ax.legend()

        ## panel 0: the shipped statistic, with all three thresholds, so a dead source gate is visible
        _panel(var_axes[0], var_gate_stat, "map max: empty vs occupied",
               [("Otsu", otsu_threshold(var_gate_stat, var_empty)[0], "--"),
                ("sup", var_gate_threshold, ":"),
                ("source", var_args.source_empty_threshold, "-.")])
        ## panel 1: the scale-invariant candidate - a raised floor lifts max but not max/mean
        _panel(var_axes[1], var_stats["max/mean"], "max/mean: empty vs occupied",
               [("Otsu", otsu_threshold(var_stats["max/mean"], var_empty)[0], "--")])
        var_axes[2].scatter(var_mag_real, var_mag_shuffled, s=6, alpha=0.4)
        var_axes[2].set_xlabel("mass real"); var_axes[2].set_ylabel("mass shuffled")
        var_axes[2].set_title("mass under input shuffle")
        var_axes[3].imshow(var_maps_real.mean(axis=0), origin="lower")
        var_axes[3].set_title("mean predicted map"); var_axes[3].axis("off")
        var_fig.tight_layout()
        var_fig.savefig(var_args.out, dpi=120)
        print(f"\nFigure: {var_args.out}")


if __name__ == "__main__":
    run()
