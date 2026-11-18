"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##

import argparse
import random 
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import *
from src.data.load_data import (load_data_y, encode_data_y, encode_occupancy_y,
                                load_wimans_room_x, wimans_room_names, wimans_room_source)
from src.data.activity_transfer import WIMANS_ACTIVITY_ALIASES
from src.utils import *
from src.data.unified_rooms import WIMANS_BOUNDS, load_wimans_room_targets, normalize_xy_meters
from src.models.room_localization import (run_localization_cross_domain,
                                          format_localization_result, localizer_class)
from configs.preset import preset

import time
import json
from datetime import datetime
from pathlib import Path

#
## Models the cross-domain runner sends down the single-target (x, y) localizer path instead of the
## classification / density path. "localizer"/"room_localizer" take the --backbone value; "that"/
## "resnet1d" are shorthand that also select their own backbone, matching the few-shot runners' names.
LOCALIZER_MODELS = ("localizer", "room_localizer", "that", "resnet1d")


def _model_room_names(var_model):
    """
    [description]
    : the room keys a run should score on. "anywhere" is a localizer-only source: its annotation
      carries continuous (x, y) positions, not the discrete locations the classification / density
      encoders read, so it is kept only for the localizer path and dropped everywhere else (where
      including it would only crash the layout lookup).
    """
    #
    var_names = wimans_room_names()
    if var_model in LOCALIZER_MODELS:
        return var_names
    return [var_name for var_name in var_names if var_name != "anywhere"]

#
##
class LazyRoomArray:
    """
    [description]
    : lazy array-like view of one room's CSI. The room is read only when the array is actually
      consumed, so the cross-domain runner never holds every room's (N, 3000, 270) float32 array
      (~3.3 MB/frame) resident at once: a training room is materialized once, when its dataset is
      built, and a test room once per repeat, inside the evaluation loop. Exposes .shape, .ndim,
      .dtype, .reshape(...), len() and np.asarray(), so it drops into the code that expects the array
      load_wimans_room_x returns.
    """
    #
    ##
    def __init__(self, var_loader, var_shape):
        self._loader = var_loader
        self._shape = tuple(var_shape)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return np.dtype(np.float32)

    def __len__(self):
        return self._shape[0]

    def __array__(self, dtype=None, copy=None):
        var_x = self._loader()
        return var_x if dtype is None else np.asarray(var_x, dtype=dtype)

    def reshape(self, *var_shape):
        return np.asarray(self).reshape(*var_shape)

    def __getitem__(self, var_index):
        return np.asarray(self)[var_index]


def _room_shape(var_data_x, var_label_list, var_length):
    """
    [description]
    : the shape LazyRoomArray will produce, without reading the whole room: the frame count and the
      window length are known from the annotation, and the trailing shape comes from reading one frame.
    """
    #
    ##
    if not var_label_list:
        return (0, var_length)
    var_probe = load_wimans_room_x(var_data_x, var_label_list[:1])
    return (len(var_label_list), var_length) + tuple(var_probe.shape[2:])


def _load_room(preset, var_task, var_model, var_users, var_env):
    """
    [description]
    : build the lazy CSI source and the labels of one room for the given model. The room key is
      resolved through wimans_room_source, so a bare environment name reads the real WiMANS source and
      a "<name><suffix>" key reads the simulated source with the same base room and its own band; the
      CSI itself is read only when the returned source is consumed.
    """
    #
    ##
    var_base, var_data_x, var_data_y, var_band = wimans_room_source(var_env)
    data_pd_y = load_data_y(var_data_y, var_environment=[var_base], var_wifi_band=var_band,
                            var_num_users=var_users)
    
    if var_task == "activity":
        data_pd_y = data_pd_y.copy()
        for var_col in (f"user_{var_i}_activity" for var_i in range(1, 7)):
            data_pd_y[var_col] = data_pd_y[var_col].replace(WIMANS_ACTIVITY_ALIASES)
    var_label_list = data_pd_y["label"].to_list()
    if not var_label_list:
        raise ValueError(
            f"no frames for room {var_env!r} (base {var_base!r}, band {var_band}, "
            f"occupants {var_users})")
    var_x = LazyRoomArray(lambda: load_wimans_room_x(var_data_x, var_label_list),
                          _room_shape(var_data_x, var_label_list, preset["data"]["length"]))
    if var_model in ("density_map", "density_map_dem"):
        ## Density-map group counting predicts per-location occupancy, so the label is the room's
        ## occupancy vector rather than a task encoding. The simulated rooms reuse their base room's
        ## layout -- same physical room -- so the layout is looked up by the base name.
        var_y = encode_occupancy_y(data_pd_y, var_base)
    else:
        var_y = encode_data_y(data_pd_y, var_task)
        if var_model in ("AMAR_WO_RVQ", "AMAR"):
            var_y = reduce_dataset(var_y, var_task, preset["nn"]["num_obj_queries"])
    return var_x, var_y


def location_position_bank():
    """
    [description]
    : {room: {class index: (x, y)}} in meters for the location task, built from
      preset["encoding"]["location"] (which class index each location letter is) and preset["layouts"]
      (where each letter sits in that room). Layout values are in units of 1000 cm, i.e. 10 m, with the
      origin at the transmitter, so multiplying by 10 puts every room in one metres-from-transmitter
      frame: that is the frame the per-room position scatter draws, and it makes the rooms comparable.
    : return: (bank, bounds), bounds = (x_min, x_max, y_min, y_max) in meters of the WiMANS rectangle
    """
    #
    ##
    var_letters = {}
    for var_name, var_vector in preset["encoding"]["location"].items():
        var_vector = np.asarray(var_vector)
        if var_vector.sum():
            ## "nan" is the all-zero no-location encoding, so every other key has its own index
            var_letters[int(var_vector.argmax())] = var_name
    var_rooms = [var_room for var_room in wimans_room_names() if var_room in preset["layouts"]]
    var_bank = {var_room: {var_index: tuple(float(var_coord) * 10.0
                                            for var_coord in preset["layouts"][var_room][var_letter])
                           for var_index, var_letter in var_letters.items()}
                for var_room in var_rooms}
    return var_bank, WIMANS_BOUNDS


def master_splitter(preset, var_task, var_model, var_users, var_train_envs):
    """
    [description]
    : build the per-room training sets and the per-room test sets. Pass one training room for the
      one-room protocol, or two for leave-one-room-out (train on two, test on the third). Every room
      not in var_train_envs becomes a test room.
    : The room list is wimans_room_names(), so with --hp data.synth_wimans.enabled=true the simulated
      rooms ("<name><suffix>") are test rooms unless they are named in var_train_envs. Each room's X is
      a LazyRoomArray, read only when the model consumes it.
    : return: train_sets_by_env, test_sets_by_env (dict {env_name: (X, y)})
    """
    var_all_envs = _model_room_names(var_model)
    for var_env in var_train_envs:
        if var_env not in var_all_envs:
            raise ValueError(f"training room {var_env!r} is not one of {var_all_envs}")
    train_sets_by_env = {var_env: _load_room(preset, var_task, var_model, var_users, var_env)
                         for var_env in var_train_envs}
    test_sets_by_env = {var_env: _load_room(preset, var_task, var_model, var_users, var_env)
                        for var_env in var_all_envs if var_env not in var_train_envs}
    return train_sets_by_env, test_sets_by_env


def _standardize_windows(var_x):
    """
    [description]
    : z-score every CSI window over its time x subcarrier entries, removing the per-device gain/AGC
      level so only the window's frequency-selective shape is left. Identical to the transform the
      few-shot runner applies, so the two protocols see the same input.
    """
    #
    var_mu = var_x.mean(axis=(1, 2), keepdims=True)
    var_sd = var_x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((var_x - var_mu) / var_sd).astype(np.float32)


def _load_localizer_room(var_env, var_features):
    """
    [description]
    : one WiMANS room as single-target (X, XY) samples for the localizer, in meters from the
      transmitter. load_wimans_room_targets keeps only the single-occupant frames (a one-position
      model cannot consume a multi-user label) and routes a "<name><suffix>" key through
      wimans_room_source, so the real, simulated and "anywhere" sources all load the same way.
    """
    #
    var_x, var_y = load_wimans_room_targets(var_env, var_length=preset["hwild"]["window"],
                                            var_antennas=var_features)
    if len(var_x) == 0:
        raise ValueError(f"no single-occupant WiMANS frames for room {var_env!r}")
    print(f"  {var_env:12s} X {var_x.shape} XY {var_y.shape} "
          f"({len(np.unique(var_y.round(3), axis=0))} distinct positions)")
    return var_x, var_y


def load_localizer_rooms(var_train_envs, var_test_envs, var_features, var_standardize):
    """
    [description]
    : load the train/test rooms for the localizer and normalise them into one shared,
      transmitter-anchored frame. This mirrors scripts/run_few_shot_multi_rooms.py (load ->
      per-window z-score -> one rectangle over every loaded position), so a normal cross-domain run on
      the same rooms and backbone is directly comparable to that runner's few-shot teacher/student.
    : return: (train_sets, test_sets, var_rect, var_spans, var_bounds)
    """
    #
    var_all_envs = wimans_room_names()
    for var_env in list(var_train_envs) + list(var_test_envs):
        if var_env not in var_all_envs:
            raise ValueError(f"room {var_env!r} is not one of {var_all_envs}")
    train_sets = {var_env: _load_localizer_room(var_env, var_features)
                  for var_env in var_train_envs}
    test_sets = {var_env: _load_localizer_room(var_env, var_features)
                 for var_env in var_test_envs}
    ## one rectangle over every loaded position covers them all; its origin is the transmitter, so
    ## every room predicts in the same metric frame and a normalized offset means the same distance
    var_all_xy = np.concatenate([var_y for _, var_y in
                                 list(train_sets.values()) + list(test_sets.values())])
    var_margin = 0.5
    var_rect = (float(var_all_xy[:, 0].min() - var_margin), float(var_all_xy[:, 0].max() + var_margin),
                float(var_all_xy[:, 1].min() - var_margin), float(var_all_xy[:, 1].max() + var_margin))
    if var_standardize == "window":
        for var_sets in (train_sets, test_sets):
            for var_room, (var_x, var_y) in var_sets.items():
                var_sets[var_room] = (_standardize_windows(var_x), var_y)
        print("CSI: per-window z-score (gain/AGC removed)")
    else:
        print("CSI: raw amplitude")
    for var_sets in (train_sets, test_sets):
        for var_room, (var_x, var_y) in var_sets.items():
            var_sets[var_room] = (var_x, normalize_xy_meters(var_y, var_rect))
    ## every room is normalised by the one rectangle, so its span (what the harness converts an error
    ## back to meters with) is the rectangle's span
    var_span = (var_rect[1] - var_rect[0], var_rect[3] - var_rect[2])
    var_all_rooms = list(train_sets) + list(test_sets)
    var_spans = {var_room: var_span for var_room in var_all_rooms}
    var_bounds = {var_room: var_rect for var_room in var_all_rooms}
    return train_sets, test_sets, var_rect, var_spans, var_bounds


def run_localizer(var_task, var_model, var_backbone, var_repeat, var_train_envs, var_features,
                  var_standardize, var_epochs):
    """
    [description]
    : the localizer cross-domain protocol: train the single-target (x, y) regressor on every
      --train_envs room and score it on every other WiMANS room (real, simulated, and the few-shot
      "anywhere" source). This is the non-few-shot counterpart of scripts/run_few_shot_multi_rooms.py:
      same rooms, same --backbone, same loading and shared frame, but trained on the full training
      rooms instead of a few-shot support fraction, so the two runs can be compared directly.
    """
    #
    if var_epochs is not None:
        preset["hwild"]["epoch"] = var_epochs
    var_test_envs = [var_env for var_env in wimans_room_names() if var_env not in var_train_envs]
    train_sets, test_sets, var_rect, var_spans, var_bounds = load_localizer_rooms(
        var_train_envs, var_test_envs, var_features, var_standardize)
    print(f"\nShared transmitter-anchored frame rectangle (m): "
          f"x [{var_rect[0]:.2f}, {var_rect[1]:.2f}], y [{var_rect[2]:.2f}, {var_rect[3]:.2f}]  "
          f"(span {var_rect[1] - var_rect[0]:.2f} x {var_rect[3] - var_rect[2]:.2f} m)")
    var_save_path = reserve_viz_dir(path=f"cross_domain/{join_envs(var_train_envs)}")
    var_results = run_localization_cross_domain(
        train_sets, test_sets, var_repeat=var_repeat, save_path=str(var_save_path),
        var_spans=var_spans, var_bounds=var_bounds, var_model_factory=localizer_class(var_backbone))
    var_result = {
        "protocol": ("train the localizer on WiMANS (real, simulated and/or anywhere) -> test the "
                     "held-out WiMANS rooms, shared transmitter-anchored metric frame"),
        "model": var_model,
        "backbone": var_backbone,
        "task": var_task,
        "train_envs": var_train_envs,
        "test_envs": var_test_envs,
        "repeat": var_repeat,
        "epochs": preset["hwild"]["epoch"],
        "window": preset["hwild"]["window"],
        "features": var_features,
        "standardize": var_standardize,
        "tx_frame_rect_m": var_rect,
        "per_env": var_results,
    }
    var_formatted = format_localization_result(var_train_envs, var_test_envs, var_results)
    var_stem = build_run_stem(f"room_localizer-{var_backbone}", "localization",
                              f"train-{join_envs(var_train_envs)}",
                              f"test-{join_envs(var_test_envs)}",
                              f"r{var_repeat}", f"e{preset['hwild']['epoch']}", run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "cross_domain", var_stem, var_result, var_formatted)
    print(var_formatted)
    print(f"\nResults saved to: {var_json_path}")
    print(f"Report saved to:  {var_txt_path}")


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--task", default = preset["task"], type = str)
    var_args.add_argument("--repeat", default = preset["repeat"], type = int)
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
    var_args.add_argument("--env", default="empty_room", type=str, help="training room name")
    var_args.add_argument("--train_envs", default=None, type=str,
                          help="Comma-separated training rooms. Overrides --env. Pass two rooms for "
                               "leave-one-room-out: train on those two and test on the remaining one. "
                               "Every room not listed becomes a test room.")
    var_args.add_argument("--backbone", default="that", choices=("that", "resnet1d"),
                          help="localizer backbone when --model is 'localizer'/'room_localizer' (it "
                               "is ignored by the classification / density models). 'that' is "
                               "RoomLocalizer over the WiMANS THAT backbone, 'resnet1d' is "
                               "ResNet1DLocalizer over the CSI window (src/models/resnet1d.py). "
                               "--model that / --model resnet1d are shorthand that also set this.")
    var_args.add_argument("--features", default="full", choices=("avg", "full"),
                          help="localizer CSI layout: 'full' (default) keeps the 3 x 3 x 30 = 270 grid "
                               "the few-shot runs use; 'avg' averages one antenna axis to 90")
    var_args.add_argument("--standardize", default="window", choices=("none", "window"),
                          help="localizer input scaling: 'window' (default) z-scores each CSI window, "
                               "removing the per-capture gain/AGC scale the simulated source shares "
                               "with the few-shot runs; 'none' feeds raw amplitude")
    var_args.add_argument("--epochs", default=None, type=int,
                          help="Override preset['nn']['epoch'] (classification / density models) or "
                               "preset['hwild']['epoch'] (localizer). The cosine LR schedule is tied "
                               "to this value, so it must match the actual training length.")
    var_args.add_argument("--hp", action="append", default=None, type=str, metavar="KEY=VALUE",
                          help="Override a preset field, dotted path, e.g. --hp nn.backbone=thatv "
                               "--hp nn.encoder_norm=group. Repeatable. Values are parsed as "
                               "bool/int/float, else kept as str.")
    #
    return var_args.parse_args()


def parse_hp_value(var_value):
    """
    [description]
    : parse the right-hand side of a --hp override into a bool / int / float / str.
    """
    var_value = var_value.strip()
    ## accept a quoted value, e.g. --hp path.save_dir='/tmp/x'
    if len(var_value) >= 2 and var_value[0] == var_value[-1] and var_value[0] in ("'", '"'):
        return var_value[1:-1]
    var_lower = var_value.lower()
    if var_lower in ("true", "false"):
        return var_lower == "true"
    if var_lower in ("none", "null"):
        return None
    try:
        return int(var_value)
    except ValueError:
        pass
    try:
        return float(var_value)
    except ValueError:
        pass
    return var_value


def apply_hp_overrides(var_overrides):
    """
    [description]
    : apply --hp overrides onto the module-level preset dict, in place, before any model is built.
    : return: dict, the applied {dotted_key: value} for the result record
    """
    var_applied = {}
    for var_item in var_overrides or []:
        var_key, var_sep, var_value = var_item.partition("=")
        if not var_sep:
            raise ValueError(f"--hp expects KEY=VALUE, got {var_item!r}")
        var_target = preset
        var_parts = var_key.strip().split(".")
        for var_part in var_parts[:-1]:
            if var_part not in var_target:
                raise KeyError(f"--hp path {var_key!r}: preset has no {var_part!r}")
            var_target = var_target[var_part]
        var_parsed = parse_hp_value(var_value)
        var_target[var_parts[-1]] = var_parsed
        var_applied[var_key.strip()] = var_parsed
    return var_applied


def format_result(var_model, var_task, result):
    """
    Build a formatted string of the results.
    """
    # (avg_key, se_key, label) in display order
    metric_specs = [
        ("avg_accuracy",    "se_accuracy",    "Avg Accuracy"),
        ("avg_precision",   "se_precision",   "Avg Precision"),
        ("avg_recall",      "se_recall",      "Avg Recall"),
        ("avg_f1_score",    "se_f1_score",    "Avg F1 Score"),
        ("avg_PPP",         "se_PPP",         "Avg Perfect Prediction %"),
        ("avg_total_error", "se_total_error", "Avg Total Error"),
    ]

    lines = ["=" * 80]

    def _const_baseline_line(var_env_results):
        ## shared by the density head (inside the mass block) and the coords head (which reports a
        ## position error without the mass metrics), so both print one identical line
        return (f"  Constant-position Baseline (m): "
                f"{var_env_results['avg_const_position_error_m']:.4f} "
                f"± {var_env_results.get('se_const_position_error_m', 0.0):.4f} (SE) "
                f"[mean training-room position; beat it to show a transferable mapping]")
    lines.append(f"EXPERIMENT RESULTS - Model: {var_model}, Task: {var_task}")
    lines.append("=" * 80)

    per_env = result.get("per_env") if isinstance(result, dict) else None

    if isinstance(per_env, dict) and per_env:
        lines.append("PER_ENV_RESULTS:")
        for env_key, env_results in per_env.items():
            lines.append(f"\n{env_key}:")
            if 'avg_mae' in env_results:
                ## density-map group-count metrics
                lines.append("  GROUP-COUNT METRICS:")
                lines.append(f"  Exact-count Accuracy: {env_results['avg_accuracy']:.4f} ± {env_results['se_accuracy']:.4f} (SE)")
                if 'avg_balanced_accuracy' in env_results:
                    lines.append(f"  Balanced-count Accuracy: {env_results['avg_balanced_accuracy']:.4f} "
                                 f"± {env_results['se_balanced_accuracy']:.4f} (SE)")
                lines.append(f"  Count MAE: {env_results['avg_mae']:.4f} ± {env_results['se_mae']:.4f} (SE)")
                lines.append(f"  Occupancy Accuracy: {env_results['avg_occupancy_accuracy']:.4f} ± {env_results['se_occupancy_accuracy']:.4f} (SE)")
                lines.append(f"  Occupancy F1: {env_results['avg_occupancy_f1']:.4f} ± {env_results['se_occupancy_f1']:.4f} (SE)")
                if 'avg_mass_accuracy' in env_results:
                    lines.append(f"  Layout-free Mass Accuracy: {env_results['avg_mass_accuracy']:.4f} "
                                 f"± {env_results['se_mass_accuracy']:.4f} (SE)")
                    lines.append(f"  Layout-free Mass MAE: {env_results['avg_mass_mae']:.4f} "
                                 f"± {env_results['se_mass_mae']:.4f} (SE)")
                    lines.append(f"  Position Error (m): {env_results['avg_position_error_m']:.4f} "
                                 f"± {env_results['se_position_error_m']:.4f} (SE)")
                    if 'avg_const_position_error_m' in env_results:
                        lines.append(_const_baseline_line(env_results))
                    lines.append(f"  Position Miss Rate: {env_results['avg_position_miss_rate']:.4f} "
                                 f"± {env_results['se_position_miss_rate']:.4f} (SE)")
                    if 'avg_position_false_positive_rate' in env_results:
                        lines.append(f"  Position False Positive Rate: "
                                     f"{env_results['avg_position_false_positive_rate']:.4f} "
                                     f"± {env_results['se_position_false_positive_rate']:.4f} (SE)")
                elif 'avg_position_error_m' in env_results:
                    ## coords head: a position error with no density mass metrics
                    lines.append(f"  Position Error (m): {env_results['avg_position_error_m']:.4f} "
                                 f"± {env_results['se_position_error_m']:.4f} (SE)")
                    if 'avg_const_position_error_m' in env_results:
                        lines.append(_const_baseline_line(env_results))
                per_class = env_results.get('per_class_accuracy', {})
                if per_class:
                    lines.append("  Per-count Accuracy: "
                                 + ", ".join(f"{k}:{v:.3f}" for k, v in per_class.items()))
                var_decision = format_decision(env_results)
                if var_decision:
                    lines.append(f"  Decision: {var_decision}")
            else:
                for avg_key, se_key, label in metric_specs:
                    if avg_key in env_results:
                        se = env_results.get(se_key, float("nan"))
                        lines.append(f"  {label}: {env_results[avg_key]:.4f} ± {se:.4f} (SE)")
    elif isinstance(result, dict):
        lines.append("SINGLE MODEL RESULTS:")
        has_aggregated = any(avg_key in result for avg_key, _, _ in metric_specs)
        if has_aggregated:
            for avg_key, se_key, label in metric_specs:
                if avg_key in result:
                    se = result.get(se_key, float("nan"))
                    lines.append(f"  {label}: {result[avg_key]:.4f} ± {se:.4f} (SE)")
        elif "precision" in result:
            # Raw single-run results (no averaging across repeats)
            single_specs = [
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("perfect_prediction_percentage", "Perfect Prediction %"),
                ("f1_score", "F1 Score"),
                ("accuracy", "Accuracy"),
                ("total_error", "Total Error"),
            ]
            for key, label in single_specs:
                if key in result:
                    lines.append(f"  {label}: {result[key]:.4f}")

    return "\n".join(lines)

#
##
def run():
    """
    [description]
    : run WiFi-based models
    """
    SEED = 103 # Ensuring the results are reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #
    ## parse arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    var_users = [u.strip() for u in var_args.users.split(',')]
    var_env = var_args.env
    var_train_envs = ([e.strip() for e in var_args.train_envs.split(',')]
                      if var_args.train_envs else [var_env])
    if var_args.epochs is not None:
        preset["nn"]["epoch"] = var_args.epochs
    var_hp_applied = apply_hp_overrides(var_args.hp)
    ## The localizer is a single-target (x, y) regressor with its own loading and metric pipeline, so
    ## it takes a separate branch here; --model that / --model resnet1d select the backbone inline,
    ## otherwise --backbone does.
    if var_model in LOCALIZER_MODELS:
        var_backbone = var_model if var_model in ("that", "resnet1d") else var_args.backbone
        run_localizer(var_task, var_model, var_backbone, var_repeat, var_train_envs,
                      var_args.features, var_args.standardize, var_args.epochs)
        return
    ## The simulated rooms reuse their base room's layout (same physical room), so alias each
    ## "<name><suffix>" key to it once the --hp overrides have enabled the source. This is what keeps
    ## the density-map models' preset["layouts"][room] lookups working for a simulated room without
    ## teaching every call site about the suffix.
    for var_room in _model_room_names(var_model):
        if var_room not in preset["layouts"]:
            var_base = wimans_room_source(var_room)[0]
            if var_base in preset["layouts"]:
                preset["layouts"][var_room] = preset["layouts"][var_base]

    # Ensuring there is no data leakage while doing splits.
    train_sets_by_env, test_sets_by_env = master_splitter(
        preset, var_task, var_model, var_users, var_train_envs)

    ## the run directory is named after the training rooms so one-room and leave-one-room-out runs
    ## do not collide. The numeric leaf is claimed with mkdir() rather than an is_dir() check
    ## followed by a create: concurrent runs (the LoR matrix starts several combos per fold at once)
    ## would otherwise both see the same leaf free, both pick it, and both write their checkpoints and
    ## figures into it. mkdir() is atomic at the OS level, so exactly one run wins a leaf.
    var_viz_root = Path(f'./visualizations/cross_domain/{"_".join(var_train_envs)}')
    var_viz_root.mkdir(parents=True, exist_ok=True)
    var_index = 1
    while True:
        save_path = var_viz_root / str(var_index)
        try:
            save_path.mkdir()
            break
        except FileExistsError:
            var_index += 1
    #
    ## run WiFi-based model
    ## The location runs also get the position scatter, which needs the metres-from-transmitter
    ## coordinates each location class index stands for (the density heads draw their own spatial
    ## figures: a density map and a coordinate heatmap).
    var_location_positions, var_location_bounds = (location_position_bank()
                                                   if var_task == "location" else (None, None))
    if var_model == "density_map":
        all_envs_results = run_density_map_cross_domain(train_sets_by_env, test_sets_by_env,
                                                        var_repeat, var_task, save_path)
    elif var_model == "density_map_dem":
        all_envs_results = run_density_map_dem_cross_domain(train_sets_by_env, test_sets_by_env,
                                                            var_repeat, var_task, save_path)
    elif var_model == "dummy_model":
        all_envs_results = run_dummy_cross_domain(train_sets_by_env, test_sets_by_env,
                                                  var_repeat, var_task, var_train_envs, save_path,
                                                  var_location_positions=var_location_positions,
                                                  var_location_bounds=var_location_bounds)
    else:
        all_envs_results = run_cross_domain(train_sets_by_env, test_sets_by_env,
                                            var_repeat, var_task, var_train_envs, save_path,
                                            var_location_positions=var_location_positions,
                                            var_location_bounds=var_location_bounds)
    #
    ##
    result = {
    "per_env": all_envs_results,
    "model": var_model,
    "task": var_task,
    "repeat": var_repeat,
    "train_envs": var_train_envs,
    "test_envs": [e for e in _model_room_names(var_model) if e not in var_train_envs],
    "epochs": preset["nn"]["epoch"],
    "data": preset["data"],
    "nn": preset["nn"],
    "hp_overrides": var_hp_applied,
    ## the metres-from-transmitter position of each location class index, per room: the input of the
    ## per-room predictions_<room>.png scatter, kept so the figure can be read without guessing where a
    ## predicted location index actually is
    "location_positions_m": ({var_room: {str(var_index): list(var_xy)
                                         for var_index, var_xy in var_bank.items()}
                              for var_room, var_bank in var_location_positions.items()}
                             if var_location_positions is not None else None),
    }

    formatted = format_result(var_model, var_task, result)
    var_test_envs = [e for e in _model_room_names(var_model) if e not in var_train_envs]
    var_stem = build_run_stem(var_model, var_task,
                              f"train-{join_envs(var_train_envs)}", f"test-{join_envs(var_test_envs)}",
                              f"r{var_repeat}", f"e{preset['nn']['epoch']}", run_timestamp())
    var_json_path, var_txt_path = save_run_outputs(
        preset["path"].get("save_dir", "output"), "cross_domain", var_stem, result, formatted)

    print(formatted)
    print(f"\nResults saved to: {var_json_path}")
    print(f"Report saved to:  {var_txt_path}")

if __name__ == "__main__":
    #
    ##
    start_time = time.time()
    run()
    print("Total time: %s seconds" % (time.time() - start_time))