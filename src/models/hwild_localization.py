"""
[file]          hwild_localization.py
[description]   Room-agnostic (x, y) localization on the H-WILD dataset.

                The CSI backbone (THAT, the same encoder the WiMANS models use) feeds a small
                regression head that outputs one normalized position per window. Positions live in
                the fixed per-room [0, 1]^2 frame of src/data/hwild.py, so a model trained on some
                rooms predicts normalized coordinates that denormalize to meters in any room.

                A direct regression head is used rather than the WiMANS density head: reusing
                DensityMapNet for continuous single-target localization collapses to the marginal map
                (the predicted peak barely moves across windows, and the model does not beat a
                constant-position baseline even on a room it was trained on). Regressing the position
                directly uses the CSI and beats that baseline.

                The leave-one-room-out runner trains on N rooms and evaluates on a held-out room. Its
                model selection and the reported error are in meters, using each room's physical span
                (HWILD_ROOMS), so a 16 m room and a 10 m room are compared on equal footing.
"""
#
##

import copy
import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs.preset import preset
from src.models.bce_that import THAT
from src.models.density_map import get_cosine_schedule_with_warmup, augment_csi
from src.utils import select_device, save_run_outputs, build_run_stem, join_envs, run_timestamp
import wandb

#
##


class HWILDLocalizer(nn.Module):
    """
    [description]
    : CSI backbone + a regression head that predicts the normalized (x, y) position of the target
      person from one CSI window. The backbone is THAT, shared with the WiMANS models; the head is a
      small MLP. The output is in the fixed per-room [0, 1]^2 frame (src/data/hwild.py).
    """
    #
    ##
    def __init__(self, var_x_shape, embedding_dim=100, hidden_dim=None, dropout=None):
        #
        super().__init__()
        if hidden_dim is None:
            hidden_dim = preset["hwild"].get("head_hidden", 128)
        if dropout is None:
            dropout = preset["hwild"].get("head_dropout", 0.2)
        #
        self.backbone = THAT(var_x_shape, [embedding_dim])
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2),
        )

    #
    ##
    def forward(self, x):
        """
        [return]
        : xy: (batch, 2) predicted normalized position in the shared per-room [0, 1]^2 frame
        """
        return self.head(self.backbone(x))


def localization_metrics(var_pred_xy, var_true_xy, var_span):
    """
    [description]
    : Euclidean localization error between predicted and true normalized positions, in meters.
    : var_pred_xy / var_true_xy: (N, 2) normalized positions
    : var_span: (x_span, y_span) in meters, or an (N, 2) per-sample array when the set mixes rooms
    : return: dict of mean / RMSE / median error in meters, and the mean normalized error
    """
    #
    var_delta = (np.asarray(var_pred_xy, dtype=np.float64) - np.asarray(var_true_xy, dtype=np.float64))
    var_span = np.asarray(var_span, dtype=np.float64)
    var_delta_m = var_delta * var_span
    var_dist = np.sqrt((var_delta_m ** 2).sum(axis=1))
    #
    return {
        "mean_error_m": float(var_dist.mean()),
        "rmse_m": float(np.sqrt((var_dist ** 2).mean())),
        "median_error_m": float(np.median(var_dist)),
        "mean_error_norm": float(np.sqrt((var_delta ** 2).sum(axis=1)).mean()),
    }


def _plot_hwild_curves(var_history, var_save_dir, var_tag="training"):
    """
    [description]
    : rewrite the training/validation loss and validation localization-error curves after an epoch.
    """
    #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    #
    os.makedirs(str(var_save_dir), exist_ok=True)
    var_fig, var_axes = plt.subplots(1, 2, figsize=(11, 4))
    var_axes[0].plot(var_history["epoch"], var_history["train_loss"], label="train")
    var_axes[0].plot(var_history["epoch"], var_history["valid_loss"], label="valid")
    var_axes[0].set_xlabel("epoch"); var_axes[0].set_ylabel("Huber loss"); var_axes[0].legend()
    var_axes[1].plot(var_history["epoch"], var_history["valid_error_m"], label="valid mean error (m)")
    var_axes[1].plot(var_history["epoch"], var_history["valid_error_m_smoothed"], label="smoothed")
    var_axes[1].set_xlabel("epoch"); var_axes[1].set_ylabel("error (m)"); var_axes[1].legend()
    var_fig.suptitle(f"{var_tag} - H-WILD localization")
    var_fig.tight_layout(rect=[0, 0, 1, 0.95])
    var_path = os.path.join(str(var_save_dir), f"{var_tag}_curves.png")
    var_fig.savefig(var_path, dpi=110)
    plt.close(var_fig)
    return var_path


def train_hwild(model, optimizer, data_train_set, data_valid_set, var_batch_size, var_epochs,
                device, var_span_bank, patience=60, var_save_dir=None, var_tag="training"):
    """
    [description]
    : train the localizer. The objective is the Huber loss between the predicted and true normalized
      positions; model selection tracks the EMA-smoothed validation mean error in meters, so the
      checkpoint is the best localizer, not the best fit to the training rooms.
    : data_train_set / data_valid_set: TensorDataset of (CSI window, normalized xy, room index). The
      room index selects the sample's physical span from var_span_bank for the validation error.
    : var_span_bank: tensor (num_rooms, 2) of (x_span, y_span) in meters, one row per training room
    : return: (best state dict, history dict)
    """
    #
    var_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    var_valid_loader = DataLoader(data_valid_set, len(data_valid_set))
    #
    var_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=preset["nn"]["scheduler"]["num_warmup_epochs"] * len(var_train_loader),
        num_training_steps=var_epochs * len(var_train_loader),
        min_lr_ratio=preset["nn"]["scheduler"]["min_lr_ratio"])
    #
    var_history = {var_key: [] for var_key in (
        "epoch", "train_loss", "valid_loss", "valid_error_m", "valid_error_m_smoothed", "learning_rate")}
    var_best_score, var_best_weight, var_epoch_saved, var_counter = np.inf, None, 0, 0
    var_ema_error, var_ema_decay = None, 0.3
    #
    for var_epoch in range(var_epochs):
        var_time = time.time()
        model.train()
        var_loss_sum, var_batches = 0.0, 0
        for var_batch in var_train_loader:
            var_x, var_xy = var_batch[0].to(device), var_batch[1].to(device)
            var_x = augment_csi(var_x)
            var_loss = F.smooth_l1_loss(model(var_x), var_xy)
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
            var_loss_sum += float(var_loss.detach())
            var_batches += 1
        var_train_loss = var_loss_sum / max(1, var_batches)
        #
        model.eval()
        with torch.no_grad():
            var_valid_batch = next(iter(var_valid_loader))
            var_valid_x, var_valid_xy = var_valid_batch[0].to(device), var_valid_batch[1].to(device)
            var_valid_loss = float(F.smooth_l1_loss(model(var_valid_x), var_valid_xy).detach())
            var_pred_xy = model(var_valid_x).cpu().numpy()
        var_span = var_span_bank[var_valid_batch[2]].numpy()
        var_metrics = localization_metrics(var_pred_xy, var_valid_xy.cpu().numpy(), var_span)
        var_ema_error = (var_metrics["mean_error_m"] if var_ema_error is None
                         else var_ema_decay * var_metrics["mean_error_m"] + (1 - var_ema_decay) * var_ema_error)
        #
        wandb.log({"epoch": var_epoch, "train_loss": var_train_loss, "valid_loss": var_valid_loss,
                   "valid_error_m": var_metrics["mean_error_m"],
                   "valid_error_m_smoothed": var_ema_error,
                   "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {var_epoch}/{var_epochs} - %.2fs" % (time.time() - var_time),
              "- Loss %.6f" % var_train_loss, "- ValidLoss %.6f" % var_valid_loss,
              "- ValidErr %.4f m" % var_metrics["mean_error_m"], "- Smooth %.4f m" % var_ema_error)
        #
        for var_key, var_value in (("epoch", var_epoch), ("train_loss", var_train_loss),
                                   ("valid_loss", var_valid_loss),
                                   ("valid_error_m", var_metrics["mean_error_m"]),
                                   ("valid_error_m_smoothed", var_ema_error),
                                   ("learning_rate", optimizer.param_groups[0]["lr"])):
            var_history[var_key].append(var_value)
        if var_save_dir is not None:
            _plot_hwild_curves(var_history, var_save_dir, var_tag)
        #
        if var_ema_error < var_best_score:
            var_best_score, var_best_weight, var_epoch_saved, var_counter = \
                var_ema_error, copy.deepcopy(model.state_dict()), var_epoch, 0
        else:
            var_counter += 1
        if var_counter >= patience:
            print(f"Early stopping triggered at epoch {var_epoch}")
            break
    #
    if var_best_weight is None:
        var_best_weight = copy.deepcopy(model.state_dict())
    print(f"Epoch that the model was saved {var_epoch_saved} - best smoothed error {var_best_score:.4f} m")
    return var_best_weight, var_history


def visualize_hwild_predictions(var_true_xy, var_pred_xy, var_room, save_dir, var_num_samples=400):
    """
    [description]
    : scatter the true and predicted positions of a test room in raw meters, so a run's spatial error
      is visible at a glance. Saves <save_dir>/hwild_predictions_<room>.png.
    : var_true_xy / var_pred_xy: (N, 2) normalized positions
    """
    #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.hwild import denormalize_xy, HWILD_ROOMS
    #
    os.makedirs(save_dir, exist_ok=True)
    var_num = min(var_num_samples, len(var_true_xy))
    var_pick = np.linspace(0, len(var_true_xy) - 1, var_num).astype(int)
    var_true_m = denormalize_xy(var_true_xy[var_pick], var_room)
    var_pred_m = denormalize_xy(var_pred_xy[var_pick], var_room)
    #
    var_fig, var_ax = plt.subplots(figsize=(6, 6))
    var_ax.scatter(var_true_m[:, 0], var_true_m[:, 1], s=10, c="tab:green", alpha=0.5, label="true")
    var_ax.scatter(var_pred_m[:, 0], var_pred_m[:, 1], s=10, c="tab:red", alpha=0.5, label="pred")
    var_x_range = HWILD_ROOMS[var_room]["x_range"]
    var_y_range = HWILD_ROOMS[var_room]["y_range"]
    var_ax.set_xlim(var_x_range); var_ax.set_ylim(var_y_range)
    var_ax.set_aspect("equal"); var_ax.set_xlabel("x (m)"); var_ax.set_ylabel("y (m)")
    var_ax.set_title(f"H-WILD {var_room} - true vs predicted positions")
    var_ax.legend(loc="upper right")
    var_path = os.path.join(save_dir, f"hwild_predictions_{var_room}.png")
    var_fig.tight_layout(); var_fig.savefig(var_path, dpi=120); plt.close(var_fig)
    return var_path


def run_hwild_cross_domain(train_sets_by_room, test_sets_by_room, var_repeat=3,
                           save_path="./visualizations/hwild"):
    """
    [description]
    : leave-one-room-out localization. Trains on every room in train_sets_by_room and evaluates on
      every room in test_sets_by_room. Each training room keeps a 90/10 split so model selection and
      the reported error are anchored on a validation set that covers every training room; the test
      rooms contribute no labels. The error is the Euclidean distance between the predicted and the
      ground-truth UWB position in meters.
    : train_sets_by_room / test_sets_by_room: dict room name -> (X (N, window, 90), XY (N, 2) normalized)
    : return: dict, room name -> averaged localization metrics with SE
    """
    #
    from src.data.hwild import room_span
    #
    device = select_device()
    print(f"Using device: {device}")
    #
    var_train_rooms = list(train_sets_by_room)
    var_room_index = {var_room: var_idx for var_idx, var_room in enumerate(var_train_rooms)}
    var_span_bank = torch.tensor([room_span(var_room) for var_room in var_train_rooms], dtype=torch.float32)
    #
    ## ------------------------------------------ data prep -------------------------------------------
    var_train_datasets, var_valid_datasets = [], []
    for var_room in var_train_rooms:
        var_x_room, var_y_room = train_sets_by_room[var_room]
        var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(len(var_x_room), var_x_room.shape[1], -1)
        var_y_room = np.asarray(var_y_room, dtype=np.float32)
        var_idx_room = np.full(len(var_x_room), var_room_index[var_room], dtype=np.int64)
        var_dataset = TensorDataset(torch.from_numpy(var_x_room), torch.from_numpy(var_y_room),
                                    torch.from_numpy(var_idx_room))
        var_perm = np.random.RandomState(39).permutation(len(var_dataset))
        var_num_valid = max(1, int(round(0.1 * len(var_dataset))))
        var_valid_idx = np.sort(var_perm[:var_num_valid])
        var_train_idx = np.sort(var_perm[var_num_valid:])
        var_train_datasets.append(torch.utils.data.Subset(var_dataset, var_train_idx.tolist()))
        var_valid_datasets.append(torch.utils.data.Subset(var_dataset, var_valid_idx.tolist()))
    #
    data_train_set = torch.utils.data.ConcatDataset(var_train_datasets)
    data_valid_set = torch.utils.data.ConcatDataset(var_valid_datasets)
    var_x_shape = tuple(var_train_datasets[0].dataset.tensors[0].shape[1:])
    print(f"Training rooms: {var_train_rooms} ({len(data_train_set)} train / "
          f"{len(data_valid_set)} valid) | Test rooms: {list(test_sets_by_room)}")
    #
    ## --------------------------------------- train & evaluate ---------------------------------------
    var_env_rep_metrics, var_env_last_xy = {}, {}
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        wandb.init(project="hwild_localization", name=f"HWILDLoc{var_r}_" + "_".join(var_train_rooms),
                   config=preset, reinit=True)
        torch.random.manual_seed(var_r + 39)
        #
        model = HWILDLocalizer(var_x_shape).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=preset["hwild"]["lr"],
                                     weight_decay=preset["hwild"]["weight_decay"])
        var_best_weight, _ = train_hwild(
            model=model, optimizer=optimizer, data_train_set=data_train_set, data_valid_set=data_valid_set,
            var_batch_size=preset["hwild"]["batch_size"], var_epochs=preset["hwild"]["epoch"],
            device=device, var_span_bank=var_span_bank, patience=preset["hwild"]["patience"],
            var_save_dir=save_path, var_tag=f"train_r{var_r}")
        model.load_state_dict(var_best_weight)
        model.eval()
        #
        for var_room, (var_x_room, var_y_room) in test_sets_by_room.items():
            var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(
                len(var_x_room), var_x_room.shape[1], -1)
            var_y_room = np.asarray(var_y_room, dtype=np.float32)
            var_loader = DataLoader(TensorDataset(torch.from_numpy(var_x_room)),
                                    batch_size=preset["hwild"]["batch_size"], shuffle=False)
            var_pred_list = []
            with torch.no_grad():
                for (var_x,) in var_loader:
                    var_pred_list.append(model(var_x.to(device)).cpu())
            var_pred_xy = torch.cat(var_pred_list, dim=0).numpy()
            var_metrics = localization_metrics(var_pred_xy, var_y_room, room_span(var_room))
            var_env_rep_metrics.setdefault(var_room, []).append(var_metrics)
            var_env_last_xy[var_room] = (var_y_room, var_pred_xy)
            #
            wandb.log({f"test_results_per_env/{var_room}/mean_error_m": var_metrics["mean_error_m"],
                       f"test_results_per_env/{var_room}/rmse_m": var_metrics["rmse_m"],
                       f"test_results_per_env/{var_room}/median_error_m": var_metrics["median_error_m"]},
                      step=var_r + 100000)
            print(f"  [{var_room}] MDE {var_metrics['mean_error_m']:.4f} m - "
                  f"RMSE {var_metrics['rmse_m']:.4f} m - Median {var_metrics['median_error_m']:.4f} m")
        #
        if var_r != var_repeat - 1:
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            var_ckpt = {"state_dict": model.state_dict(), "x_shape": var_x_shape,
                        "train_rooms": var_train_rooms}
            torch.save(var_ckpt, os.path.join(save_path, "hwild_model.pth"))
    wandb.finish()
    #
    ## ---------------------------------------- aggregate ---------------------------------------------
    var_metric_names = ("mean_error_m", "rmse_m", "median_error_m", "mean_error_norm")
    results = {}
    for var_room, var_rep_list in var_env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list])
            var_std = float(var_arr.std(ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"avg_{var_name}"] = float(var_arr.mean())
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        results[var_room] = var_env_result
        var_true_xy, var_pred_xy = var_env_last_xy[var_room]
        visualize_hwild_predictions(var_true_xy, var_pred_xy, var_room,
                                    os.path.join(save_path, var_room))
        print(f"\n[{var_room}] avg over {var_repeat} repeats: "
              f"MDE {var_env_result['avg_mean_error_m']:.4f} ± {var_env_result['se_mean_error_m']:.4f} m | "
              f"RMSE {var_env_result['avg_rmse_m']:.4f} ± {var_env_result['se_rmse_m']:.4f} m")
    #
    return results


def load_hwild_model(var_path, device=None):
    """
    [description]
    : rebuild an HWILDLocalizer saved by run_hwild_cross_domain and load its weights, so a live or new
      capture can be run through it.
    : return: (model, checkpoint dict)
    """
    #
    var_device = device if device is not None else select_device()
    var_ckpt = torch.load(var_path, map_location=var_device)
    var_model = HWILDLocalizer(tuple(var_ckpt["x_shape"])).to(var_device)
    var_model.load_state_dict(var_ckpt["state_dict"])
    var_model.eval()
    return var_model, var_ckpt


def predict_hwild_xy(var_model, var_x, var_device=None):
    """
    [description]
    : run one CSI window (or a batch) through a loaded H-WILD model and return the predicted
      normalized (x, y). Denormalize with src.data.hwild.denormalize_xy(..., room) for meters.
    : var_x: numpy array (window, 90) or (N, window, 90)
    : return: numpy array (2,) or (N, 2) normalized positions
    """
    #
    var_device = var_device if var_device is not None else next(var_model.parameters()).device
    var_single = var_x.ndim == 2
    var_batch = var_x[None] if var_single else var_x
    with torch.no_grad():
        var_xy = var_model(torch.from_numpy(np.asarray(var_batch, dtype=np.float32)).to(var_device)).cpu().numpy()
    return var_xy[0] if var_single else var_xy


def format_hwild_result(var_train_rooms, var_test_rooms, results):
    """
    [description]
    : human-readable report of a leave-one-room-out H-WILD localization run.
    """
    #
    var_lines = ["=" * 80,
                 f"H-WILD ROOM-AGNOSTIC LOCALIZATION - train {join_envs(var_train_rooms)}, "
                 f"test {join_envs(var_test_rooms)}",
                 "=" * 80]
    for var_room, var_env_result in results.items():
        var_lines.append(f"\n{var_room}:")
        var_lines.append(f"  Mean Distance Error: {var_env_result['avg_mean_error_m']:.4f} "
                         f"± {var_env_result['se_mean_error_m']:.4f} m (SE)")
        var_lines.append(f"  RMSE:                {var_env_result['avg_rmse_m']:.4f} "
                         f"± {var_env_result['se_rmse_m']:.4f} m (SE)")
        var_lines.append(f"  Median Error:        {var_env_result['avg_median_error_m']:.4f} "
                         f"± {var_env_result['se_median_error_m']:.4f} m (SE)")
    return "\n".join(var_lines)
