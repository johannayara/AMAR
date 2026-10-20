"""
[file]          room_localization.py
[description]   Room-agnostic single-target (x, y) localization from CSI, over any set of rooms.

                The CSI backbone (THAT, the encoder the WiMANS models use) feeds a small regression
                head that outputs one normalized position per window. Every room is expressed in its
                own [0, 1]^2 frame (src/data/hwild.py for H-WILD, src/data/unified_rooms.py for the
                combined WiMANS + H-WILD set), so one model trained on some rooms predicts normalized
                coordinates that denormalize to meters in any room.

                A direct regression head is used rather than the WiMANS density head: reusing
                DensityMapNet for continuous single-target localization collapses to the marginal map
                (the predicted position barely moves across windows, and the model does not beat a
                constant-position baseline even on a room it was trained on). Regressing the position
                directly uses the CSI and beats that baseline.

                The leave-one-room-out runner trains on N rooms and evaluates on a held-out room. Its
                model selection and the reported error are in meters, using each room's physical span,
                so rooms of different sizes are compared on equal footing. Every report also carries a
                constant-position baseline (the mean training-room position), the reference a
                room-agnostic model must beat to show a transferable CSI-to-position mapping.
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
from src.utils import (select_device, save_run_outputs, build_run_stem, join_envs, run_timestamp,
                       visualize_room_grid_performance)
import wandb

#
##


class RoomLocalizer(nn.Module):
    """
    [description]
    : CSI backbone + a regression head that predicts the normalized (x, y) position of the target
      person from one CSI window. The backbone is THAT, shared with the WiMANS models; the head is a
      small MLP. The output is in the room's [0, 1]^2 frame.
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
        : xy: (batch, 2) predicted normalized position in the room's [0, 1]^2 frame
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


def _resolve_spans(var_rooms, var_spans):
    """
    [description]
    : per-room (x_span, y_span) in meters. Defaults to the H-WILD geometry when not given.
    """
    if var_spans is not None:
        return var_spans
    from src.data.hwild import room_span
    return {var_room: room_span(var_room) for var_room in var_rooms}


def _resolve_bounds(var_rooms, var_bounds):
    """
    [description]
    : per-room (x_min, x_max, y_min, y_max) in meters. Defaults to the H-WILD geometry when not given.
    """
    if var_bounds is not None:
        return var_bounds
    from src.data.hwild import HWILD_ROOMS
    return {var_room: (HWILD_ROOMS[var_room]["x_range"][0], HWILD_ROOMS[var_room]["x_range"][1],
                       HWILD_ROOMS[var_room]["y_range"][0], HWILD_ROOMS[var_room]["y_range"][1])
            for var_room in var_rooms}


def _plot_curves(var_history, var_save_dir, var_tag="training"):
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
    var_fig.suptitle(f"{var_tag} - room localization")
    var_fig.tight_layout(rect=[0, 0, 1, 0.95])
    var_path = os.path.join(str(var_save_dir), f"{var_tag}_curves.png")
    var_fig.savefig(var_path, dpi=110)
    plt.close(var_fig)
    return var_path


def train_room_localizer(model, optimizer, data_train_set, data_valid_set, var_batch_size, var_epochs,
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
            _plot_curves(var_history, var_save_dir, var_tag)
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


def visualize_localization_predictions(var_true_xy, var_pred_xy, var_room, save_dir, var_bounds,
                                       var_num_samples=400):
    """
    [description]
    : scatter the true and predicted positions of a test room in raw meters, so a run's spatial error
      is visible at a glance. Saves <save_dir>/predictions_<room>.png.
    : var_true_xy / var_pred_xy: (N, 2) normalized positions
    : var_bounds: (x_min, x_max, y_min, y_max) in meters of the room's frame
    """
    #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    #
    os.makedirs(save_dir, exist_ok=True)
    var_x0, var_x1, var_y0, var_y1 = var_bounds
    var_num = min(var_num_samples, len(var_true_xy))
    var_pick = np.linspace(0, len(var_true_xy) - 1, var_num).astype(int)
    var_true_xy = np.asarray(var_true_xy)[var_pick]
    var_pred_xy = np.asarray(var_pred_xy)[var_pick]
    var_true_m = np.stack([var_x0 + var_true_xy[:, 0] * (var_x1 - var_x0),
                           var_y0 + var_true_xy[:, 1] * (var_y1 - var_y0)], axis=1)
    var_pred_m = np.stack([var_x0 + var_pred_xy[:, 0] * (var_x1 - var_x0),
                           var_y0 + var_pred_xy[:, 1] * (var_y1 - var_y0)], axis=1)
    #
    var_fig, var_ax = plt.subplots(figsize=(6, 6))
    var_ax.scatter(var_true_m[:, 0], var_true_m[:, 1], s=10, c="tab:green", alpha=0.5, label="true")
    var_ax.scatter(var_pred_m[:, 0], var_pred_m[:, 1], s=10, c="tab:red", alpha=0.5, label="pred")
    var_ax.set_xlim(var_x0, var_x1); var_ax.set_ylim(var_y0, var_y1)
    var_ax.set_aspect("equal"); var_ax.set_xlabel("x (m)"); var_ax.set_ylabel("y (m)")
    var_ax.set_title(f"{var_room} - true vs predicted positions")
    var_ax.legend(loc="upper right")
    var_path = os.path.join(save_dir, f"predictions_{var_room}.png")
    var_fig.tight_layout(); var_fig.savefig(var_path, dpi=120); plt.close(var_fig)
    return var_path


def run_localization_cross_domain(train_sets_by_room, test_sets_by_room, var_repeat=3,
                                  save_path="./visualizations/room_localization",
                                  var_spans=None, var_bounds=None):
    """
    [description]
    : leave-one-room-out localization over an arbitrary room set (possibly mixing datasets). Trains on
      every room in train_sets_by_room and evaluates on every room in test_sets_by_room. Each training
      room keeps a 90/10 split so model selection and the reported error are anchored on a validation
      set that covers every training room; the test rooms contribute no labels. The error is the
      Euclidean distance between the predicted and the ground-truth position in meters.
    : train_sets_by_room / test_sets_by_room: dict room name -> (X (N, window, features), XY (N, 2) normalized)
    : var_spans: dict room -> (x_span, y_span) in meters (defaults to H-WILD geometry)
    : var_bounds: dict room -> (x_min, x_max, y_min, y_max) in meters (defaults to H-WILD geometry)
    : return: dict, room name -> averaged localization metrics with SE
    """
    #
    device = select_device()
    print(f"Using device: {device}")
    #
    var_all_rooms = list(train_sets_by_room) + list(test_sets_by_room)
    var_spans = _resolve_spans(var_all_rooms, var_spans)
    var_bounds = _resolve_bounds(var_all_rooms, var_bounds)
    var_train_rooms = list(train_sets_by_room)
    var_room_index = {var_room: var_idx for var_idx, var_room in enumerate(var_train_rooms)}
    var_span_bank = torch.tensor([var_spans[var_room] for var_room in var_train_rooms], dtype=torch.float32)
    #
    ## ------------------------------------------ data prep -------------------------------------------
    var_train_datasets, var_valid_datasets = [], []
    var_train_y_all = []
    for var_room in var_train_rooms:
        var_x_room, var_y_room = train_sets_by_room[var_room]
        var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(len(var_x_room), var_x_room.shape[1], -1)
        var_y_room = np.asarray(var_y_room, dtype=np.float32)
        var_train_y_all.append(var_y_room)
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
    ## The mean training-room position is a trivial "no-information" localizer. It is reported next
    ## to every test-room error: a model that does not beat it has not learned a transferable mapping
    ## from CSI to position, however low the raw error looks.
    var_train_mean_xy = np.concatenate(var_train_y_all).mean(axis=0)
    print(f"Training rooms: {var_train_rooms} ({len(data_train_set)} train / "
          f"{len(data_valid_set)} valid) | Test rooms: {list(test_sets_by_room)}")
    print(f"Constant-position baseline (mean training position): {var_train_mean_xy.round(3)}")
    #
    ## --------------------------------------- train & evaluate ---------------------------------------
    var_env_rep_metrics, var_env_last_xy = {}, {}
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        wandb.init(project="room_localization", name=f"RoomLoc{var_r}_" + "_".join(var_train_rooms),
                   config=preset, reinit=True)
        torch.random.manual_seed(var_r + 39)
        #
        model = RoomLocalizer(var_x_shape).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=preset["hwild"]["lr"],
                                     weight_decay=preset["hwild"]["weight_decay"])
        var_best_weight, _ = train_room_localizer(
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
            var_metrics = localization_metrics(var_pred_xy, var_y_room, var_spans[var_room])
            var_metrics["const_mean_error_m"] = localization_metrics(
                np.tile(var_train_mean_xy, (len(var_y_room), 1)), var_y_room,
                var_spans[var_room])["mean_error_m"]
            var_env_rep_metrics.setdefault(var_room, []).append(var_metrics)
            var_env_last_xy[var_room] = (var_y_room, var_pred_xy)
            #
            wandb.log({f"test_results_per_env/{var_room}/mean_error_m": var_metrics["mean_error_m"],
                       f"test_results_per_env/{var_room}/rmse_m": var_metrics["rmse_m"],
                       f"test_results_per_env/{var_room}/median_error_m": var_metrics["median_error_m"]},
                      step=var_r + 100000)
            print(f"  [{var_room}] MDE {var_metrics['mean_error_m']:.4f} m - "
                  f"RMSE {var_metrics['rmse_m']:.4f} m - Median {var_metrics['median_error_m']:.4f} m - "
                  f"const {var_metrics['const_mean_error_m']:.4f} m")
        #
        if var_r != var_repeat - 1:
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            var_ckpt = {"state_dict": model.state_dict(), "x_shape": var_x_shape,
                        "train_rooms": var_train_rooms}
            torch.save(var_ckpt, os.path.join(save_path, "model.pth"))
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
        var_env_result["avg_const_mean_error_m"] = float(
            np.mean([var_rep["const_mean_error_m"] for var_rep in var_rep_list]))
        results[var_room] = var_env_result
        var_true_xy, var_pred_xy = var_env_last_xy[var_room]
        visualize_localization_predictions(var_true_xy, var_pred_xy, var_room,
                                           os.path.join(save_path, var_room), var_bounds[var_room])
        ## the standard per-location figures, with the room's grid cells as locations
        visualize_room_grid_performance([[var_t] for var_t in var_true_xy],
                                        [[var_p] for var_p in var_pred_xy],
                                        os.path.join(save_path, var_room),
                                        var_grid=preset["viz"]["grid"])
        print(f"\n[{var_room}] avg over {var_repeat} repeats: "
              f"MDE {var_env_result['avg_mean_error_m']:.4f} ± {var_env_result['se_mean_error_m']:.4f} m | "
              f"RMSE {var_env_result['avg_rmse_m']:.4f} ± {var_env_result['se_rmse_m']:.4f} m | "
              f"constant-baseline MDE {var_env_result['avg_const_mean_error_m']:.4f} m")
    #
    return results


#
## ---------------------------------------------------------------------------------------------- ##
## ---------------------------------- few-shot distillation ------------------------------------- ##
#
## H-WILD -> WiMANS few-shot: a teacher RoomLocalizer is trained on H-WILD, frozen, and regularises a
## student that is fine-tuned on a small fraction of the WiMANS frames. The student is then scored on
## the WiMANS frames neither the support set nor its own validation slice saw. This mirrors the repo's
## few-shot harness (run_AMAR_WO_RVQ_few_shot, run_density_map_few_shot); the structural difference is
## that RoomLocalizer is a single-target regressor, so the distillation term is a plain smooth-L1 match
## of the two outputs -- there is no query order to resolve, unlike SetDistillationLoss.
#
##

def train_student_localizer(model, teacher, optimizer, data_train_set, data_valid_set, var_batch_size,
                            var_epochs, device, var_span, var_kd_weight=1.0, patience=60,
                            var_save_dir=None, var_tag="student"):
    """
    [description]
    : fine-tune a RoomLocalizer student on a few-shot support set. The objective is the supervised
      smooth-L1 position loss on the support plus var_kd_weight times a smooth-L1 match to the frozen
      teacher's output on the same support windows. Model selection tracks the EMA-smoothed validation
      mean error in meters, exactly as train_room_localizer does.
    : model / teacher: RoomLocalizer; teacher must be in eval() and is used under no_grad
    : data_train_set / data_valid_set: TensorDataset of (CSI window, normalized position)
    : var_span: (x_span, y_span) of the shared frame in meters, used for the reported error
    : var_kd_weight: 0 disables distillation (the pure few-shot ablation)
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
    var_history = {var_key: [] for var_key in ("epoch", "train_loss", "valid_loss", "supervised_loss",
                                               "kd_loss", "valid_error_m", "valid_error_m_smoothed")}
    var_best_score, var_best_weight, var_epoch_saved, var_counter = np.inf, None, 0, 0
    var_ema_error, var_ema_decay = None, 0.3
    #
    for var_epoch in range(var_epochs):
        model.train()
        var_loss_sum, var_sup_sum, var_kd_sum, var_batches = 0.0, 0.0, 0.0, 0
        for var_batch in var_train_loader:
            var_x, var_xy = var_batch[0].to(device), var_batch[1].to(device)
            var_x = augment_csi(var_x)
            var_pred = model(var_x)
            var_sup = F.smooth_l1_loss(var_pred, var_xy)
            var_kd = var_pred.new_zeros(())
            if var_kd_weight > 0:
                with torch.no_grad():
                    var_teacher_pred = teacher(var_x)
                var_kd = F.smooth_l1_loss(var_pred, var_teacher_pred)
            var_loss = var_sup + var_kd_weight * var_kd
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            var_scheduler.step()
            var_loss_sum += float(var_loss.detach())
            var_sup_sum += float(var_sup.detach())
            var_kd_sum += float(var_kd.detach())
            var_batches += 1
        #
        model.eval()
        with torch.no_grad():
            var_valid_batch = next(iter(var_valid_loader))
            var_valid_x, var_valid_xy = var_valid_batch[0].to(device), var_valid_batch[1].to(device)
            var_valid_loss = float(F.smooth_l1_loss(model(var_valid_x), var_valid_xy).detach())
            var_pred_xy = model(var_valid_x).cpu().numpy()
        var_metrics = localization_metrics(var_pred_xy, var_valid_xy.cpu().numpy(), var_span)
        var_ema_error = (var_metrics["mean_error_m"] if var_ema_error is None
                         else var_ema_decay * var_metrics["mean_error_m"] + (1 - var_ema_decay) * var_ema_error)
        for var_key, var_value in (("epoch", var_epoch),
                                   ("train_loss", var_loss_sum / max(1, var_batches)),
                                   ("valid_loss", var_valid_loss),
                                   ("supervised_loss", var_sup_sum / max(1, var_batches)),
                                   ("kd_loss", var_kd_sum / max(1, var_batches)),
                                   ("valid_error_m", var_metrics["mean_error_m"]),
                                   ("valid_error_m_smoothed", var_ema_error)):
            var_history[var_key].append(var_value)
        print(f"Epoch {var_epoch}/{var_epochs} - Loss %.6f (sup %.6f, kd %.6f) - ValidErr %.4f m - "
              "Smooth %.4f m" % (var_history["train_loss"][-1], var_history["supervised_loss"][-1],
                                 var_history["kd_loss"][-1], var_metrics["mean_error_m"], var_ema_error))
        if var_save_dir is not None:
            _plot_curves(var_history, var_save_dir, var_tag)
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
    print(f"Epoch that the student was saved {var_epoch_saved} - best smoothed error {var_best_score:.4f} m")
    return var_best_weight, var_history


def run_few_shot_localization(hwild_sets_by_room, wimans_sets_by_room, var_few_shot_ratio=0.05,
                              var_kd_weight=1.0, var_repeat=3, var_teacher_epochs=None,
                              var_student_epochs=None, var_init_from_teacher=True, var_bounds=None,
                              save_path="./visualizations/few_shot_localization"):
    """
    [description]
    : H-WILD -> WiMANS few-shot localization by knowledge distillation.
        teacher  -- RoomLocalizer trained on every H-WILD room, then frozen.
        student  -- RoomLocalizer initialised from the teacher and fine-tuned on var_few_shot_ratio of
                    the WiMANS pool with supervised Huber + var_kd_weight * a smooth-L1 match to the
                    frozen teacher on the same support windows.
        test     -- the WiMANS frames held out of both the support set and the student's validation
                    slice, so the reported error is on frames neither model was tuned on.
      The teacher is reported on the same query frames, next to the constant-position baseline, so the
      few-shot gain over the teacher and over a trivial predictor is visible.
      Note the teacher's own WiMANS error is worse than the constant baseline (H-WILD's device-link
      channel does not transfer to WiMANS passive scattering), so the distillation term pulls the
      student toward weak targets; var_kd_weight=0 gives the pure few-shot ablation and is the honest
      reference. See the report for both.
    : hwild_sets_by_room: dict room -> (X, Y) normalized in the shared frame; the teacher's data
    : wimans_sets_by_room: dict room -> (X, Y) normalized in the shared frame; the support/valid/query
      pool
    : var_few_shot_ratio: fraction of the WiMANS pool used to fine-tune the student
    : var_kd_weight: weight of the distillation term (0 = supervised few-shot only)
    : var_teacher_epochs / var_student_epochs: default preset["hwild"]["epoch"]
    : var_init_from_teacher: warm-start the student from the teacher's weights (standard KD init)
    : var_bounds: (x0, x1, y0, y1) of the shared frame in meters
    : return: dict with per-room student/teacher/baseline metrics
    """
    #
    device = select_device()
    print(f"Using device: {device}")
    if var_teacher_epochs is None:
        var_teacher_epochs = preset["hwild"]["epoch"]
    if var_student_epochs is None:
        var_student_epochs = preset["hwild"]["epoch"]
    if var_bounds is None:
        raise ValueError("run_few_shot_localization needs var_bounds (the shared frame rectangle)")
    var_span = (var_bounds[1] - var_bounds[0], var_bounds[3] - var_bounds[2])
    var_span_bank = torch.tensor([var_span], dtype=torch.float32)
    var_wimans_rooms = list(wimans_sets_by_room)
    #
    ## ------------------------------------ teacher data (H-WILD) ------------------------------------
    var_teacher_x = np.concatenate([np.asarray(hwild_sets_by_room[var_room][0], dtype=np.float32)
                                    for var_room in hwild_sets_by_room])
    var_teacher_y = np.concatenate([np.asarray(hwild_sets_by_room[var_room][1], dtype=np.float32)
                                    for var_room in hwild_sets_by_room])
    var_teacher_x = var_teacher_x.reshape(len(var_teacher_x), var_teacher_x.shape[1], -1)
    var_x_shape = tuple(var_teacher_x.shape[1:])
    var_t_perm = np.random.RandomState(39).permutation(len(var_teacher_x))
    var_t_nvalid = max(1, int(round(0.1 * len(var_teacher_x))))
    var_t_valid = np.sort(var_t_perm[:var_t_nvalid])
    var_t_train = np.sort(var_t_perm[var_t_nvalid:])
    var_t_idx = np.zeros(len(var_teacher_x), dtype=np.int64)
    var_teacher_train_set = TensorDataset(torch.from_numpy(var_teacher_x[var_t_train]),
                                          torch.from_numpy(var_teacher_y[var_t_train]),
                                          torch.from_numpy(var_t_idx[var_t_train]))
    var_teacher_valid_set = TensorDataset(torch.from_numpy(var_teacher_x[var_t_valid]),
                                          torch.from_numpy(var_teacher_y[var_t_valid]),
                                          torch.from_numpy(var_t_idx[var_t_valid]))
    #
    ## ------------------------- WiMANS pool: support / validation / query ---------------------------
    var_w_x = np.concatenate([np.asarray(wimans_sets_by_room[var_room][0], dtype=np.float32)
                              for var_room in var_wimans_rooms])
    var_w_y = np.concatenate([np.asarray(wimans_sets_by_room[var_room][1], dtype=np.float32)
                              for var_room in var_wimans_rooms])
    var_w_room = np.concatenate([np.full(len(wimans_sets_by_room[var_room][0]), var_idx, dtype=np.int64)
                                 for var_idx, var_room in enumerate(var_wimans_rooms)])
    var_w_x = var_w_x.reshape(len(var_w_x), var_w_x.shape[1], -1)
    var_num = len(var_w_x)
    var_num_few = max(1, int(round(var_few_shot_ratio * var_num)))
    if var_num_few >= var_num - 1:
        raise ValueError(f"var_few_shot_ratio={var_few_shot_ratio} leaves no query data "
                         f"({var_num_few}/{var_num}). Lower the ratio or provide more WiMANS frames.")
    var_perm = np.random.RandomState(39).permutation(var_num)
    var_few_idx = np.sort(var_perm[:var_num_few])
    var_rest_idx = np.sort(var_perm[var_num_few:])
    var_num_valid = max(1, int(round(0.1 * len(var_rest_idx))))
    var_valid_idx = var_rest_idx[:var_num_valid]
    var_query_idx = var_rest_idx[var_num_valid:]
    var_student_train_set = TensorDataset(torch.from_numpy(var_w_x[var_few_idx]),
                                          torch.from_numpy(var_w_y[var_few_idx]))
    var_student_valid_set = TensorDataset(torch.from_numpy(var_w_x[var_valid_idx]),
                                          torch.from_numpy(var_w_y[var_valid_idx]))
    print(f"H-WILD teacher: {len(var_teacher_x)} windows ({len(var_t_train)} train / "
          f"{len(var_teacher_valid_set)} valid)")
    print(f"WiMANS pool: {var_num} frames -> support {var_num_few} ({var_few_shot_ratio:.2%}) | "
          f"student validation {var_num_valid} | query {len(var_query_idx)}")
    print(f"kd_weight {var_kd_weight} | init_from_teacher {var_init_from_teacher}")
    #
    ## ------------------------------------------ teacher --------------------------------------------
    print("\n--- teacher (H-WILD) ---")
    wandb.init(project="few_shot_localization", name=f"teacher_kd{var_kd_weight}", config=preset,
               reinit=True)
    torch.random.manual_seed(39)
    var_teacher = RoomLocalizer(var_x_shape).to(device)
    var_teacher_opt = torch.optim.Adam(var_teacher.parameters(), lr=preset["hwild"]["lr"],
                                       weight_decay=preset["hwild"]["weight_decay"])
    var_teacher_weight, _ = train_room_localizer(
        model=var_teacher, optimizer=var_teacher_opt, data_train_set=var_teacher_train_set,
        data_valid_set=var_teacher_valid_set, var_batch_size=preset["hwild"]["batch_size"],
        var_epochs=var_teacher_epochs, device=device, var_span_bank=var_span_bank,
        patience=preset["hwild"]["patience"], var_save_dir=save_path, var_tag="teacher")
    var_teacher.load_state_dict(var_teacher_weight)
    var_teacher.eval()
    for var_param in var_teacher.parameters():
        var_param.requires_grad_(False)
    wandb.finish()
    #
    ## --------------------------------------- student & test ----------------------------------------
    var_student_query_metrics, var_teacher_query_metrics = {}, {}
    ## last repeat's query predictions per room, kept for the prediction figures, as in
    ## run_localization_cross_domain
    var_env_last_xy = {}
    var_const_baseline = var_w_y[var_few_idx].mean(axis=0)
    for var_r in range(var_repeat):
        print(f"\n--- student repeat {var_r} ---")
        wandb.init(project="few_shot_localization",
                   name=f"FewShotLoc{var_r}_kd{var_kd_weight}", config=preset, reinit=True)
        torch.random.manual_seed(var_r + 39)
        var_student = RoomLocalizer(var_x_shape).to(device)
        if var_init_from_teacher:
            var_student.load_state_dict(var_teacher.state_dict())
        var_student_opt = torch.optim.Adam(var_student.parameters(), lr=preset["hwild"]["lr"],
                                           weight_decay=preset["hwild"]["weight_decay"])
        var_student_weight, _ = train_student_localizer(
            model=var_student, teacher=var_teacher, optimizer=var_student_opt,
            data_train_set=var_student_train_set, data_valid_set=var_student_valid_set,
            var_batch_size=preset["hwild"]["batch_size"], var_epochs=var_student_epochs,
            device=device, var_span=var_span, var_kd_weight=var_kd_weight,
            patience=preset["hwild"]["patience"], var_save_dir=save_path, var_tag=f"student_r{var_r}")
        var_student.load_state_dict(var_student_weight)
        var_student.eval()
        #
        for var_room_idx, var_room in enumerate(var_wimans_rooms):
            var_sel = var_query_idx[var_w_room[var_query_idx] == var_room_idx]
            if len(var_sel) == 0:
                continue
            var_q_x = torch.from_numpy(var_w_x[var_sel]).to(device)
            var_q_y = var_w_y[var_sel]
            with torch.no_grad():
                var_student_xy = var_student(var_q_x).cpu().numpy()
                var_teacher_xy = var_teacher(var_q_x).cpu().numpy()
            var_student_metrics = localization_metrics(var_student_xy, var_q_y, var_span)
            var_const_metrics = localization_metrics(
                np.tile(var_const_baseline, (len(var_q_y), 1)), var_q_y, var_span)
            var_student_metrics["const_mean_error_m"] = var_const_metrics["mean_error_m"]
            var_student_metrics["const_rmse_m"] = var_const_metrics["rmse_m"]
            var_student_metrics["const_median_error_m"] = var_const_metrics["median_error_m"]
            var_student_query_metrics.setdefault(var_room, []).append(var_student_metrics)
            var_teacher_query_metrics.setdefault(var_room, []).append(
                localization_metrics(var_teacher_xy, var_q_y, var_span))
            var_env_last_xy[var_room] = (var_q_y, var_student_xy)
            wandb.log({f"query/{var_room}/student_mean_error_m": var_student_metrics["mean_error_m"],
                       f"query/{var_room}/teacher_mean_error_m":
                           var_teacher_query_metrics[var_room][-1]["mean_error_m"],
                       f"query/{var_room}/const_mean_error_m": var_student_metrics["const_mean_error_m"]},
                      step=var_r + 100000)
            print(f"  [{var_room}] query {len(var_sel)} frames | student {var_student_metrics['mean_error_m']:.4f} m "
                  f"| teacher {var_teacher_query_metrics[var_room][-1]['mean_error_m']:.4f} m "
                  f"| const {var_student_metrics['const_mean_error_m']:.4f} m")
        wandb.finish()
        if var_r != var_repeat - 1:
            del var_student, var_student_opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    #
    ## ----------------------------------------- aggregate -------------------------------------------
    results = {}
    for var_room in var_student_query_metrics:
        var_student_list = var_student_query_metrics[var_room]
        var_teacher_list = var_teacher_query_metrics[var_room]
        var_env_result = {}
        for var_name in ("mean_error_m", "rmse_m", "median_error_m"):
            var_student_arr = np.array([var_rep[var_name] for var_rep in var_student_list])
            var_teacher_arr = np.array([var_rep[var_name] for var_rep in var_teacher_list])
            var_env_result[f"avg_student_{var_name}"] = float(var_student_arr.mean())
            var_env_result[f"se_student_{var_name}"] = (
                float(var_student_arr.std(ddof=1) / np.sqrt(len(var_student_arr)))
                if len(var_student_arr) > 1 else 0.0)
            var_env_result[f"avg_teacher_{var_name}"] = float(var_teacher_arr.mean())
            var_env_result[f"se_teacher_{var_name}"] = (
                float(var_teacher_arr.std(ddof=1) / np.sqrt(len(var_teacher_arr)))
                if len(var_teacher_arr) > 1 else 0.0)
        for var_name in ("mean_error_m", "rmse_m", "median_error_m"):
            var_env_result[f"avg_const_{var_name}"] = float(
                np.mean([var_rep[f"const_{var_name}"] for var_rep in var_student_list]))
        results[var_room] = var_env_result
        var_true_xy, var_pred_xy = var_env_last_xy[var_room]
        visualize_localization_predictions(var_true_xy, var_pred_xy, var_room,
                                           os.path.join(save_path, var_room), var_bounds)
        ## the standard per-location figures, with the room's grid cells as locations
        visualize_room_grid_performance([[var_t] for var_t in var_true_xy],
                                        [[var_p] for var_p in var_pred_xy],
                                        os.path.join(save_path, var_room),
                                        var_grid=preset["viz"]["grid"])
        print(f"\n[{var_room}] student {var_env_result['avg_student_mean_error_m']:.4f} ± "
              f"{var_env_result['se_student_mean_error_m']:.4f} m | teacher "
              f"{var_env_result['avg_teacher_mean_error_m']:.4f} m | constant baseline "
              f"{var_env_result['avg_const_mean_error_m']:.4f} m")
    #
    return results


def format_few_shot_result(var_few_shot_ratio, var_kd_weight, results):
    """
    [description]
    : human-readable report of the H-WILD -> WiMANS few-shot run.
    """
    #
    var_lines = ["=" * 80,
                 f"H-WILD -> WiMANS FEW-SHOT LOCALIZATION (support {var_few_shot_ratio:.2%}, "
                 f"kd_weight {var_kd_weight})",
                 "=" * 80]
    var_metric_labels = (("mean_error_m", "MDE"), ("rmse_m", "RMSE"), ("median_error_m", "Median"))
    for var_room, var_env_result in results.items():
        var_lines.append(f"\n{var_room}:")
        var_student = " | ".join(
            f"{var_label} {var_env_result[f'avg_student_{var_name}']:.4f} ± "
            f"{var_env_result[f'se_student_{var_name}']:.4f} m (SE)"
            for var_name, var_label in var_metric_labels)
        var_lines.append(f"  Student   {var_student}")
        var_teacher = " | ".join(
            f"{var_label} {var_env_result[f'avg_teacher_{var_name}']:.4f} ± "
            f"{var_env_result[f'se_teacher_{var_name}']:.4f} m (SE)"
            for var_name, var_label in var_metric_labels)
        var_lines.append(f"  Teacher   {var_teacher}  (H-WILD teacher, no WiMANS fine-tuning)")
        var_const = " | ".join(
            f"{var_label} {var_env_result[f'avg_const_{var_name}']:.4f} m"
            for var_name, var_label in var_metric_labels)
        var_lines.append(f"  Constant  {var_const}  (mean support position; beat this to show the "
                         f"few-shot step learned anything)")
    return "\n".join(var_lines)


def load_room_model(var_path, device=None):
    """
    [description]
    : rebuild a RoomLocalizer saved by run_localization_cross_domain and load its weights.
    : return: (model, checkpoint dict)
    """
    #
    var_device = device if device is not None else select_device()
    var_ckpt = torch.load(var_path, map_location=var_device)
    var_model = RoomLocalizer(tuple(var_ckpt["x_shape"])).to(var_device)
    var_model.load_state_dict(var_ckpt["state_dict"])
    var_model.eval()
    return var_model, var_ckpt


def predict_room_xy(var_model, var_x, var_device=None):
    """
    [description]
    : run one CSI window (or a batch) through a loaded model and return the predicted normalized
      (x, y). Denormalize with the room's bounds for meters.
    : var_x: numpy array (window, features) or (N, window, features)
    : return: numpy array (2,) or (N, 2) normalized positions
    """
    #
    var_device = var_device if var_device is not None else next(var_model.parameters()).device
    var_single = var_x.ndim == 2
    var_batch = var_x[None] if var_single else var_x
    with torch.no_grad():
        var_xy = var_model(torch.from_numpy(np.asarray(var_batch, dtype=np.float32)).to(var_device)).cpu().numpy()
    return var_xy[0] if var_single else var_xy


def format_localization_result(var_train_rooms, var_test_rooms, results):
    """
    [description]
    : human-readable report of a leave-one-room-out localization run.
    """
    #
    var_lines = ["=" * 80,
                 f"ROOM-AGNOSTIC LOCALIZATION - train {join_envs(var_train_rooms)}, "
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
        var_lines.append(f"  Constant-baseline:   {var_env_result['avg_const_mean_error_m']:.4f} m "
                         f"(mean training position; beat this to show a transferable mapping)")
    return "\n".join(var_lines)
