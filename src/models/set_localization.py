"""
[file]          set_localization.py
[description]   General multi-user room-agnostic localization by set prediction.

                The model emits a fixed set of NUM_QUERIES position hypotheses, each with an
                objectness logit, and is trained with a Hungarian assignment to the ground-truth
                position set. This is the AMAR set-prediction formulation applied to localization.
                It is "general" in the two senses the density head is not:

                  - It needs no per-room location knowledge. WiMANS' five discrete locations and
                    H-WILD's continuous positions are both just sets, so one model and one evaluation
                    cover every room of either dataset.
                  - Count and location are linked by construction: the predicted count is the number
                    of queries that assert an object, and each of those queries is one of the matched
                    positions. There is no separate peak detector whose threshold can over- or
                    under-count independently of where the mass sits.

                The Hungarian loss also removes the marginal-map solution the density head collapsed
                into: every query must commit to a position, so a query cannot hedge by lighting up
                all of a room's locations at once.
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
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

from configs.preset import preset
from src.models.bce_that import THAT
from src.models.density_map import get_cosine_schedule_with_warmup, augment_csi
from src.models.room_density import (set_localization_metrics, visualize_room_density, _resolve_spans,
                                     _resolve_bounds, _true_sets)
from src.utils import select_device, save_run_outputs, build_run_stem, join_envs, run_timestamp
import wandb

#
##


class SetLocalizer(nn.Module):
    """
    [description]
    : CSI backbone + a transformer decoder with learnable queries. Each query predicts a normalized
      (x, y) position and an objectness logit; the number of queries that assert an object is the
      predicted count.
    """
    #
    ##
    def __init__(self, var_x_shape, embedding_dim=100, num_queries=6, d_model=100, nhead=4,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        #
        super().__init__()
        self.backbone = THAT(var_x_shape, [embedding_dim])
        self.memory_proj = nn.Linear(embedding_dim, d_model)
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        var_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(var_layer, num_decoder_layers)
        self.xy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(), nn.Linear(d_model, 2))
        self.obj_head = nn.Linear(d_model, 1)
        self.num_queries = num_queries

    #
    ##
    def forward(self, x):
        """
        [return]
        : xy: (batch, num_queries, 2) normalized position hypotheses in the room's [0, 1]^2 frame
        : obj: (batch, num_queries) objectness logits
        """
        var_feat = self.backbone(x)
        var_memory = self.memory_proj(var_feat).unsqueeze(1)
        var_queries = self.query_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        var_out = self.decoder(var_queries, var_memory)
        return torch.sigmoid(self.xy_head(var_out)), self.obj_head(var_out).squeeze(-1)


def hungarian_set_loss(var_xy, var_obj, var_pos, var_mask, var_xy_weight=5.0):
    """
    [description]
    : Hungarian (set) loss. For each sample the NUM_QUERIES hypotheses are assigned to the true
      positions with a linear-sum assignment minimising L1 position distance minus objectness; matched
      queries regress their position and are trained to assert an object, unmatched queries to assert
      none. Samples with an empty true set (empty room) supervise every query as no-object.
    : var_xy: (batch, num_queries, 2) predicted normalized positions
    : var_obj: (batch, num_queries) objectness logits
    : var_pos: (batch, max_users, 2) true normalized positions
    : var_mask: (batch, max_users) 0/1 validity
    : return: scalar loss
    """
    #
    var_batch, var_queries = var_xy.shape[0], var_xy.shape[1]
    var_xy_sum = var_xy.new_zeros(())
    var_obj_target = var_xy.new_zeros((var_batch, var_queries))
    for var_idx in range(var_batch):
        var_true = var_pos[var_idx][var_mask[var_idx] > 0.5]
        if len(var_true) == 0:
            continue
        var_cost = var_xy_weight * torch.cdist(var_xy[var_idx], var_true, p=1) \
            - torch.sigmoid(var_obj[var_idx]).unsqueeze(1)
        var_rows, var_cols = linear_sum_assignment(var_cost.detach().cpu().numpy())
        var_xy_sum = var_xy_sum + F.l1_loss(var_xy[var_idx][var_rows], var_true[var_cols], reduction="sum")
        var_obj_target[var_idx, var_rows] = 1.0
    var_xy_loss = var_xy_sum / max(1, var_batch)
    var_obj_loss = F.binary_cross_entropy_with_logits(var_obj, var_obj_target)
    return var_xy_loss + var_obj_loss


def select_sets(var_xy, var_obj, var_threshold):
    """
    [description]
    : predicted position sets: the queries whose objectness probability exceeds var_threshold.
    : var_xy: (N, num_queries, 2), var_obj: (N, num_queries) logits
    : return: list of (num_selected, 2) numpy arrays
    """
    #
    var_xy = var_xy.detach().cpu().numpy() if torch.is_tensor(var_xy) else np.asarray(var_xy)
    var_prob = (torch.sigmoid(var_obj).detach().cpu().numpy() if torch.is_tensor(var_obj)
                else 1.0 / (1.0 + np.exp(-np.asarray(var_obj))))
    return [var_xy[var_idx][var_prob[var_idx] > var_threshold] for var_idx in range(len(var_xy))]


def calibrate_objectness_threshold(var_xy, var_obj, var_true_sets, var_span, var_grid=np.linspace(0.1, 0.9, 17)):
    """
    [description]
    : pick the objectness threshold that minimises the count MAE on a validation split, so the count
      decision is calibrated rather than fixed.
    : return: (best_threshold, best_count_mae)
    """
    #
    var_best_threshold, var_best_mae = float(var_grid[0]), np.inf
    for var_threshold in var_grid:
        var_sets = select_sets(var_xy, var_obj, float(var_threshold))
        var_mae = set_localization_metrics(var_sets, var_true_sets, var_span)["count_mae"]
        if var_mae < var_best_mae:
            var_best_mae, var_best_threshold = float(var_mae), float(var_threshold)
    return var_best_threshold, var_best_mae


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
    var_axes[0].set_xlabel("epoch"); var_axes[0].set_ylabel("set loss"); var_axes[0].legend()
    var_axes[1].plot(var_history["epoch"], var_history["valid_error_m"], label="valid set error (m)")
    var_axes[1].plot(var_history["epoch"], var_history["valid_error_m_smoothed"], label="smoothed")
    var_axes[1].set_xlabel("epoch"); var_axes[1].set_ylabel("error (m)"); var_axes[1].legend()
    var_fig.suptitle(f"{var_tag} - set-prediction localization")
    var_fig.tight_layout(rect=[0, 0, 1, 0.95])
    var_path = os.path.join(str(var_save_dir), f"{var_tag}_curves.png")
    var_fig.savefig(var_path, dpi=110)
    plt.close(var_fig)
    return var_path


def train_set_localizer(model, optimizer, data_train_set, data_valid_set, var_batch_size, var_epochs,
                        device, var_span_bank, patience=60, var_save_dir=None, var_tag="training"):
    """
    [description]
    : train the set localizer. The objective is the Hungarian set loss; model selection tracks the
      EMA-smoothed validation set error in meters at a fixed objectness threshold of 0.5.
    : data_train_set / data_valid_set: TensorDataset of (CSI window, positions, mask, room index)
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
            var_x = augment_csi(var_batch[0].to(device))
            var_pos, var_mask = var_batch[1].to(device), var_batch[2].to(device)
            var_xy, var_obj = model(var_x)
            var_loss = hungarian_set_loss(var_xy, var_obj, var_pos, var_mask)
            optimizer.zero_grad()
            var_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            var_scheduler.step()
            var_loss_sum += float(var_loss.detach())
            var_batches += 1
        var_train_loss = var_loss_sum / max(1, var_batches)
        #
        model.eval()
        with torch.no_grad():
            var_valid_batch = next(iter(var_valid_loader))
            var_valid_xy, var_valid_obj = model(var_valid_batch[0].to(device))
            var_valid_loss = float(hungarian_set_loss(
                var_valid_xy, var_valid_obj, var_valid_batch[1].to(device),
                var_valid_batch[2].to(device)).detach())
        var_span = var_span_bank[var_valid_batch[3]].numpy()
        var_metrics = set_localization_metrics(
            select_sets(var_valid_xy, var_valid_obj, 0.5),
            _true_sets(var_valid_batch[1], var_valid_batch[2]), var_span)
        var_ema_error = (var_metrics["mean_error_m"] if var_ema_error is None
                         else var_ema_decay * var_metrics["mean_error_m"] + (1 - var_ema_decay) * var_ema_error)
        #
        wandb.log({"epoch": var_epoch, "train_loss": var_train_loss, "valid_loss": var_valid_loss,
                   "valid_error_m": var_metrics["mean_error_m"],
                   "valid_error_m_smoothed": var_ema_error,
                   "valid_count_mae": var_metrics["count_mae"],
                   "valid_detection_f1": var_metrics["detection_f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {var_epoch}/{var_epochs} - %.2fs" % (time.time() - var_time),
              "- Loss %.6f" % var_train_loss, "- ValidLoss %.6f" % var_valid_loss,
              "- ValidErr %.4f m" % var_metrics["mean_error_m"],
              "- CntMAE %.3f" % var_metrics["count_mae"],
              "- DetF1 %.3f" % var_metrics["detection_f1"], "- Smooth %.4f m" % var_ema_error)
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


def run_set_localization_cross_domain(train_sets_by_room, test_sets_by_room, var_repeat=3,
                                      save_path="./visualizations/set_localization",
                                      var_spans=None, var_bounds=None):
    """
    [description]
    : multi-user leave-one-room-out set-prediction localization over an arbitrary room set (possibly
      mixing datasets). Trains on every room in train_sets_by_room and evaluates on every room in
      test_sets_by_room. Each training room keeps a 90/10 split; the objectness threshold and model
      selection are anchored on that validation split, and the test rooms contribute no labels.
    : train_sets_by_room / test_sets_by_room: dict room -> (X, POS (N, K, 2), MASK (N, K))
    : return: dict, room name -> averaged multi-user localization metrics with SE
    """
    #
    device = select_device()
    print(f"Using device: {device}")
    var_all_rooms = list(train_sets_by_room) + list(test_sets_by_room)
    var_spans = _resolve_spans(var_all_rooms, var_spans)
    var_bounds = _resolve_bounds(var_all_rooms, var_bounds)
    var_train_rooms = list(train_sets_by_room)
    var_room_index = {var_room: var_idx for var_idx, var_room in enumerate(var_train_rooms)}
    var_span_bank = torch.tensor([var_spans[var_room] for var_room in var_train_rooms], dtype=torch.float32)
    #
    ## ------------------------------------------ data prep -------------------------------------------
    var_train_datasets, var_valid_datasets, var_train_pos = [], [], []
    for var_room in var_train_rooms:
        var_x_room, var_pos_room, var_mask_room = train_sets_by_room[var_room]
        var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(len(var_x_room), var_x_room.shape[1], -1)
        var_pos_room = np.asarray(var_pos_room, dtype=np.float32)
        var_mask_room = np.asarray(var_mask_room, dtype=np.float32)
        var_idx_room = np.full(len(var_x_room), var_room_index[var_room], dtype=np.int64)
        var_dataset = TensorDataset(torch.from_numpy(var_x_room), torch.from_numpy(var_pos_room),
                                    torch.from_numpy(var_mask_room), torch.from_numpy(var_idx_room))
        var_perm = np.random.RandomState(39).permutation(len(var_dataset))
        var_num_valid = max(1, int(round(0.1 * len(var_dataset))))
        var_valid_idx = np.sort(var_perm[:var_num_valid])
        var_train_idx = np.sort(var_perm[var_num_valid:])
        var_train_datasets.append(torch.utils.data.Subset(var_dataset, var_train_idx.tolist()))
        var_valid_datasets.append(torch.utils.data.Subset(var_dataset, var_valid_idx.tolist()))
        var_train_pos.append(var_pos_room[var_mask_room > 0.5])
    #
    data_train_set = torch.utils.data.ConcatDataset(var_train_datasets)
    data_valid_set = torch.utils.data.ConcatDataset(var_valid_datasets)
    var_x_shape = tuple(var_train_datasets[0].dataset.tensors[0].shape[1:])
    var_train_mean_xy = np.concatenate(var_train_pos).mean(axis=0)
    print(f"Training rooms: {var_train_rooms} ({len(data_train_set)} train / "
          f"{len(data_valid_set)} valid) | Test rooms: {list(test_sets_by_room)}")
    #
    ## --------------------------------------- train & evaluate ---------------------------------------
    var_env_rep_metrics, var_env_last_sets = {}, {}
    for var_r in range(var_repeat):
        print("Repeat", var_r)
        wandb.init(project="set_localization", name=f"SetLoc{var_r}_" + "_".join(var_train_rooms),
                   config=preset, reinit=True)
        torch.random.manual_seed(var_r + 39)
        #
        model = SetLocalizer(var_x_shape, num_queries=preset["nn"]["num_obj_queries"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=preset["nn"]["lr"],
                                     weight_decay=preset["nn"]["weight_decay"])
        var_best_weight, _ = train_set_localizer(
            model=model, optimizer=optimizer, data_train_set=data_train_set, data_valid_set=data_valid_set,
            var_batch_size=preset["nn"]["batch_size"], var_epochs=preset["nn"]["epoch"], device=device,
            var_span_bank=var_span_bank, patience=preset["density"].get("patience", 60),
            var_save_dir=save_path, var_tag=f"train_r{var_r}")
        model.load_state_dict(var_best_weight)
        model.eval()
        #
        ## calibrate the objectness threshold on the training rooms' validation split
        with torch.no_grad():
            var_valid_batch = next(iter(DataLoader(data_valid_set, len(data_valid_set))))
            var_valid_xy, var_valid_obj = model(var_valid_batch[0].to(device))
        var_threshold, _ = calibrate_objectness_threshold(
            var_valid_xy, var_valid_obj, _true_sets(var_valid_batch[1], var_valid_batch[2]),
            var_span_bank[var_valid_batch[3]].numpy())
        print(f"  calibrated objectness threshold: {var_threshold:.2f}")
        #
        for var_room, (var_x_room, var_pos_room, var_mask_room) in test_sets_by_room.items():
            var_x_room = np.asarray(var_x_room, dtype=np.float32).reshape(
                len(var_x_room), var_x_room.shape[1], -1)
            var_loader = DataLoader(TensorDataset(torch.from_numpy(var_x_room)),
                                    batch_size=preset["nn"]["batch_size"], shuffle=False)
            var_xy_list, var_obj_list = [], []
            with torch.no_grad():
                for (var_x,) in var_loader:
                    var_xy, var_obj = model(var_x.to(device))
                    var_xy_list.append(var_xy.cpu()); var_obj_list.append(var_obj.cpu())
            var_pred_xy = torch.cat(var_xy_list); var_pred_obj = torch.cat(var_obj_list)
            var_pred_sets = select_sets(var_pred_xy, var_pred_obj, var_threshold)
            var_true_sets = [np.asarray(var_pos_room[i])[np.asarray(var_mask_room[i]) > 0.5]
                             for i in range(len(var_pos_room))]
            var_metrics = set_localization_metrics(var_pred_sets, var_true_sets, var_spans[var_room])
            var_metrics["const_mean_error_m"] = set_localization_metrics(
                [np.tile(var_train_mean_xy, (1, 1)) for _ in range(len(var_true_sets))],
                var_true_sets, var_spans[var_room])["mean_error_m"]
            var_metrics["objectness_threshold"] = var_threshold
            var_env_rep_metrics.setdefault(var_room, []).append(var_metrics)
            var_env_last_sets[var_room] = (var_true_sets, var_pred_sets)
            #
            wandb.log({f"test_results_per_env/{var_room}/mean_error_m": var_metrics["mean_error_m"],
                       f"test_results_per_env/{var_room}/count_mae": var_metrics["count_mae"],
                       f"test_results_per_env/{var_room}/detection_f1": var_metrics["detection_f1"]},
                      step=var_r + 100000)
            print(f"  [{var_room}] setErr {var_metrics['mean_error_m']:.4f} m - "
                  f"matchedMDE {var_metrics['mde_matched_m']:.4f} m - "
                  f"cntMAE {var_metrics['count_mae']:.3f} - exactCnt {var_metrics['exact_count_acc']:.3f} - "
                  f"detF1 {var_metrics['detection_f1']:.3f} - const {var_metrics['const_mean_error_m']:.4f} m")
        #
        if var_r != var_repeat - 1:
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            torch.save({"state_dict": model.state_dict(), "x_shape": var_x_shape,
                        "num_queries": preset["nn"]["num_obj_queries"], "train_rooms": var_train_rooms},
                       os.path.join(save_path, "model.pth"))
    wandb.finish()
    #
    ## ---------------------------------------- aggregate ---------------------------------------------
    var_metric_names = ("mean_error_m", "mde_matched_m", "count_mae", "exact_count_acc",
                        "detection_precision", "detection_recall", "detection_f1")
    results = {}
    for var_room, var_rep_list in var_env_rep_metrics.items():
        var_env_result = {}
        for var_name in var_metric_names:
            var_arr = np.array([var_rep[var_name] for var_rep in var_rep_list], dtype=np.float64)
            var_std = float(np.nanstd(var_arr, ddof=1)) if len(var_arr) > 1 else 0.0
            var_env_result[f"avg_{var_name}"] = float(np.nanmean(var_arr))
            var_env_result[f"std_{var_name}"] = var_std
            var_env_result[f"se_{var_name}"] = var_std / np.sqrt(len(var_arr)) if len(var_arr) > 1 else 0.0
        var_env_result["avg_const_mean_error_m"] = float(
            np.mean([var_rep["const_mean_error_m"] for var_rep in var_rep_list]))
        var_env_result["avg_objectness_threshold"] = float(
            np.mean([var_rep["objectness_threshold"] for var_rep in var_rep_list]))
        results[var_room] = var_env_result
        var_true_sets, var_pred_sets = var_env_last_sets[var_room]
        visualize_room_density(var_true_sets, var_pred_sets, var_room, os.path.join(save_path, var_room),
                               var_bounds[var_room])
        print(f"\n[{var_room}] avg over {var_repeat} repeats: "
              f"setErr {var_env_result['avg_mean_error_m']:.4f} ± {var_env_result['se_mean_error_m']:.4f} m | "
              f"matchedMDE {var_env_result['avg_mde_matched_m']:.4f} m | "
              f"cntMAE {var_env_result['avg_count_mae']:.3f} | "
              f"detF1 {var_env_result['avg_detection_f1']:.3f} | "
              f"constant-baseline {var_env_result['avg_const_mean_error_m']:.4f} m")
    #
    return results


def format_set_result(var_train_rooms, var_test_rooms, results):
    """
    [description]
    : human-readable report of a multi-user leave-one-room-out set-prediction run.
    """
    #
    var_lines = ["=" * 88,
                 f"MULTI-USER ROOM-AGNOSTIC LOCALIZATION (set prediction) - train "
                 f"{join_envs(var_train_rooms)}, test {join_envs(var_test_rooms)}",
                 "=" * 88]
    for var_room, var_res in results.items():
        var_lines.append(f"\n{var_room}:")
        var_lines.append(f"  Set error (m):        {var_res['avg_mean_error_m']:.4f} "
                         f"± {var_res['se_mean_error_m']:.4f} m (SE)")
        var_lines.append(f"  Matched MDE (m):      {var_res['avg_mde_matched_m']:.4f} "
                         f"± {var_res['se_mde_matched_m']:.4f} m (SE)")
        var_lines.append(f"  Count MAE:            {var_res['avg_count_mae']:.4f} "
                         f"± {var_res['se_count_mae']:.4f} (SE)")
        var_lines.append(f"  Exact-count accuracy: {var_res['avg_exact_count_acc']:.4f}")
        var_lines.append(f"  Detection P/R/F1:      {var_res['avg_detection_precision']:.3f} / "
                         f"{var_res['avg_detection_recall']:.3f} / {var_res['avg_detection_f1']:.3f}")
        var_lines.append(f"  Constant-baseline:    {var_res['avg_const_mean_error_m']:.4f} m "
                         f"(mean training position; beat this to show a transferable mapping)")
    return "\n".join(var_lines)
