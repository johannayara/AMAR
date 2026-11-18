"""
[file]          diagnose_localizer_fit.py
[description]   Answer "is the localizer failing to learn, or is the CSI uninformative?" for one room,
                without a full few-shot run.

                Loads the room with the same loader and window the localization runners use, splits it
                once with a fixed seed, and prints the mean error in meters of:

                  constant    always predict the mean training position
                  centroid    nearest per-position centroid of the per-window mean CSI (the trivial
                              baseline a network must beat, and the one that exposed the THAT
                              backbone's collapse to the mean)
                  <backbone>  the actual network, briefly trained end to end on the split

                It also prints the standard deviation of the untrained network's output across windows:
                if that is ~0 the backbone is input-invariant before any training, and training can then
                only learn a constant.

                Positions are one of the room's discrete layout points, so the centroid baseline is the
                honest yardstick: the network should be able to reach it.

                A "<room><suffix>" room name (e.g. meeting_room_synth) turns the simulated source on by
                itself, so nothing else is needed to diagnose the synthetic set.

                Example (synthetic 3 x 3 source, both backbones, one room):
                    python scripts/test_scripts/diagnose_localizer_fit.py \
                        --room meeting_room_synth --backbone resnet1d --steps 400
                    python scripts/test_scripts/diagnose_localizer_fit.py \
                        --room meeting_room_synth --backbone that --steps 400
"""
#
##

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from configs.preset import preset
from src.data.unified_rooms import load_wimans_room_targets, wimans_room_names
from src.models.room_localization import localizer_class
from src.utils import select_device

#
##


def parse_args():
    """
    [description]
    : command line arguments for the localizer fit diagnostic
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--room", default="meeting_room_synth", type=str,
                          help="a key of wimans_room_names()")
    var_args.add_argument("--backbone", default="resnet1d", choices=("resnet1d", "that"),
                          help="network to train; 'that' is RoomLocalizer's WiMANS backbone")
    var_args.add_argument("--features", default="full", choices=("avg", "full"),
                          help="as scripts/run_few_shot_multi_rooms.py: 270 (full 3 x 3 grid) or 90")
    var_args.add_argument("--synth", action="store_true",
                          help="force preset['data']['synth_wimans'] on. A '<room><suffix>' --room "
                               "enables it by itself, so this is normally unnecessary")
    var_args.add_argument("--max_samples", default=None, type=int)
    var_args.add_argument("--steps", default=400, type=int, help="gradient steps")
    var_args.add_argument("--lr", default=1e-3, type=float)
    var_args.add_argument("--batch_size", default=32, type=int)
    var_args.add_argument("--valid_frac", default=0.2, type=float)
    var_args.add_argument("--seed", default=39, type=int)
    #
    return var_args.parse_args()


def centroid_predict(var_x, var_y, var_train, var_index):
    """
    [description]
    : predicted positions of a nearest-per-position-centroid rule. Each training sample votes on the
      centroid of its position in the per-window z-scored time-mean CSI; a query sample takes the
      position of its nearest centroid.
    : return: (len(var_index), 2) float32 predicted positions in the same frame as var_y
    """
    #
    var_feat = var_x.mean(axis=1).astype(np.float64)
    var_feat = (var_feat - var_feat.mean(axis=1, keepdims=True)) / (var_feat.std(axis=1, keepdims=True) + 1e-6)
    var_key = np.round(var_y, 3)
    var_centroids, var_coords = [], []
    for var_label in sorted({tuple(var_row) for var_row in var_key[var_train]}):
        var_mask = np.array([tuple(var_row) == var_label for var_row in var_key[var_train]])
        var_centroids.append(var_feat[var_train][var_mask].mean(axis=0))
        var_coords.append(np.asarray(var_label, dtype=np.float32))
    var_centroids = np.stack(var_centroids)
    var_coords = np.stack(var_coords)
    var_dist = ((var_feat[var_index][:, None, :] - var_centroids[None]) ** 2).sum(axis=2)
    return var_coords[var_dist.argmin(axis=1)]


def centroid_error(var_x, var_y, var_train, var_valid):
    """
    [description]
    : mean error of the nearest-per-position-centroid rule (see centroid_predict).
    : return: float, mean error in meters
    """
    #
    var_pred = centroid_predict(var_x, var_y, var_train, var_valid)
    return float(np.sqrt(((var_pred - var_y[var_valid]) ** 2).sum(axis=1)).mean())


def predict(var_model, var_x, var_index):
    """
    [description]
    : the model's predicted positions on the rows of var_index, as numpy.
    """
    var_model.eval()
    with torch.no_grad():
        return var_model(var_x[var_index]).cpu().numpy()


def position_confusion(var_true_xy, var_pred_xy, var_max_labels=16):
    """
    [description]
    : the per-location confusion of a predictor over a room's discrete positions. Both the truth and
      the prediction are snapped to the nearest true position, and the (true, predicted) counts are
      tabulated. This is the readout that separates "the data does not separate the positions" from
      "the objective/optimizer does not use the signal it has": a non-trivial rule (nearest centroid)
      and the trained model are run through the same table, so their diagonals are comparable.
    : var_true_xy / var_pred_xy: (N, 2) positions in the same frame (meters)
    : var_max_labels: skip rooms with more positions than this (e.g. the "anywhere" room has hundreds)
    : return: (labels (L, 2), confusion (L, L) int, accuracy float) or None when there are too many
      distinct positions
    """
    #
    var_true_xy = np.asarray(var_true_xy, dtype=np.float64).reshape(-1, 2)
    var_pred_xy = np.asarray(var_pred_xy, dtype=np.float64).reshape(-1, 2)
    var_labels = np.unique(np.round(var_true_xy, 3), axis=0)
    if len(var_labels) > var_max_labels:
        return None
    ## snap truth and prediction to the nearest true position
    var_true_idx = ((var_true_xy[:, None, :] - var_labels[None]) ** 2).sum(axis=2).argmin(axis=1)
    var_pred_idx = ((var_pred_xy[:, None, :] - var_labels[None]) ** 2).sum(axis=2).argmin(axis=1)
    var_confusion = np.zeros((len(var_labels), len(var_labels)), dtype=int)
    for var_t, var_p in zip(var_true_idx, var_pred_idx):
        var_confusion[var_t, var_p] += 1
    var_accuracy = float(np.trace(var_confusion) / max(1, var_confusion.sum()))
    return var_labels, var_confusion, var_accuracy


def report_confusion(var_name, var_labels, var_confusion, var_accuracy):
    """
    [description]
    : print one predictor's per-location confusion and diagonal accuracy.
    """
    #
    print(f"  {var_name}: per-location accuracy {var_accuracy:.3f} over {len(var_labels)} positions "
          f"({var_confusion.sum()} frames)")
    var_header = "      true\\pred  " + " ".join(f"{var_i:5d}" for var_i in range(len(var_labels)))
    print(var_header)
    for var_t in range(len(var_labels)):
        print(f"      {var_t:9d}  " + " ".join(f"{var_confusion[var_t, var_p]:5d}"
                                               for var_p in range(len(var_labels))))


def evaluate(var_model, var_x, var_y, var_index):
    """
    [description]
    : mean error in meters of the model's predictions on the rows of var_index.
    """
    var_model.eval()
    with torch.no_grad():
        var_pred = var_model(var_x[var_index])
    return float(torch.sqrt(((var_pred - var_y[var_index]) ** 2).sum(dim=1)).mean())


##
def enable_synth_for_room(var_room):
    """
    [description]
    : a "<room><suffix>" key only exists while the simulated source is enabled, so asking for one turns
      it on rather than failing. A bare real-room name leaves the preset as it is.
    : return: bool, whether the simulated source ends up enabled
    """
    var_synth = preset["data"]["synth_wimans"]
    if var_room.endswith(var_synth.get("suffix", "_synth")):
        var_synth["enabled"] = True
    return bool(var_synth.get("enabled"))


def run():
    """
    [description]
    : run the diagnostic
    """
    var_args = parse_args()
    if var_args.synth:
        preset["data"]["synth_wimans"]["enabled"] = True
    enable_synth_for_room(var_args.room)
    if var_args.room not in wimans_room_names():
        raise ValueError(f"unknown room {var_args.room!r}; expected one of {wimans_room_names()}")

    var_x, var_y = load_wimans_room_targets(var_args.room, var_max_samples=var_args.max_samples,
                                            var_length=preset["hwild"]["window"],
                                            var_antennas=var_args.features)
    print(f"{var_args.room}: X {tuple(var_x.shape)} | "
          f"{len(np.unique(np.round(var_y, 3), axis=0))} distinct positions | "
          f"span {np.ptp(var_y[:, 0]):.2f} x {np.ptp(var_y[:, 1]):.2f} m")

    var_order = np.random.RandomState(var_args.seed).permutation(len(var_x))
    var_nvalid = max(1, int(round(var_args.valid_frac * len(var_x))))
    var_valid, var_train = np.sort(var_order[:var_nvalid]), np.sort(var_order[var_nvalid:])

    print(f"constant baseline      : "
          f"{np.sqrt(((var_y[var_valid] - var_y[var_train].mean(axis=0)) ** 2).sum(axis=1)).mean():.3f} m")
    print(f"centroid baseline      : {centroid_error(var_x, var_y, var_train, var_valid):.3f} m")

    var_device = select_device()
    torch.manual_seed(var_args.seed)
    var_class = localizer_class(var_args.backbone)
    var_model = var_class(var_x.shape[1:]).to(var_device)

    var_xt = torch.from_numpy(var_x).to(var_device)
    var_yt = torch.from_numpy(var_y).to(var_device)
    var_train_t = torch.from_numpy(var_train.astype(np.int64)).to(var_device)
    var_valid_t = torch.from_numpy(var_valid.astype(np.int64)).to(var_device)

    var_model.eval()
    with torch.no_grad():
        var_out = var_model(var_xt[:var_nvalid]).cpu().numpy()
    var_spread = var_out.std(axis=0).max()
    print(f"untrained output std   : {np.round(var_out.std(axis=0), 5)}  "
          f"(input dependence: {'present' if var_spread > 1e-4 else 'NONE - backbone is input-invariant'})")

    var_model.train()
    var_opt = torch.optim.Adam(var_model.parameters(), lr=var_args.lr)
    var_batch = min(var_args.batch_size, len(var_train))
    var_start = time.time()
    for var_step in range(var_args.steps):
        var_idx = torch.randint(len(var_train), (var_batch,), device=var_device)
        var_loss = torch.nn.functional.smooth_l1_loss(var_model(var_xt[var_train_t][var_idx]),
                                                      var_yt[var_train_t][var_idx])
        var_opt.zero_grad()
        var_loss.backward()
        var_opt.step()
        if var_step % max(1, var_args.steps // 10) == 0 or var_step == var_args.steps - 1:
            print(f"  step {var_step:4d}  loss {var_loss.item():.4f}  "
                  f"train {evaluate(var_model, var_xt, var_yt, var_train_t):.3f} m  "
                  f"valid {evaluate(var_model, var_xt, var_yt, var_valid_t):.3f} m")
            var_model.train()
    print(f"trained {var_args.backbone} for {var_args.steps} steps ({time.time() - var_start:.1f}s) "
          f"on {len(var_train)} train / {len(var_valid)} valid windows")
    #
    ## per-location confusion on the held-out frames: the nearest-centroid row says whether the room's
    ## discrete positions are separable from the CSI at all (data), the trained-model row says whether
    ## the model recovered that signal (objective/optimizer/backbone). Read the two diagonals together.
    var_valid_true = var_y[var_valid]
    var_valid_centroid = centroid_predict(var_x, var_y, var_train, var_valid)
    var_valid_model = predict(var_model, var_xt, var_valid_t)
    print(f"\nper-location confusion on {len(var_valid)} held-out windows "
          f"(rows true, cols predicted; positions are the true coordinates, in meters):")
    for var_name, var_pred in (("centroid  (data separability)", var_valid_centroid),
                               (f"{var_args.backbone:9s} (trained model)", var_valid_model)):
        var_report = position_confusion(var_valid_true, var_pred)
        if var_report is None:
            print(f"  {var_name}: more distinct positions than the confusion matrix supports")
            continue
        report_confusion(var_name, *var_report)
    print(f"  positions (m): " + " ".join(f"({var_p[0]:.2f},{var_p[1]:.2f})"
                                          for var_p in np.unique(np.round(var_valid_true, 3), axis=0)))


if __name__ == "__main__":
    #
    var_start = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - var_start))
