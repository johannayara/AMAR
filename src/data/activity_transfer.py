"""
[file]          activity_transfer.py
[description]   The shared-activity task between NTU-Fi HAR and WiMANS.

                NTU-Fi HAR labels 6 activities (box, circle, clean, fall, run, walk) and WiMANS labels
                9 (nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down, stand_up), so a
                cross-dataset activity run is only well defined on the vocabulary both datasets share.
                A run therefore restricts both sides to those shared activities, maps them onto one
                class ordering, and appends the AMAR_WO_RVQ no-object class -- so the model's output
                space is identical on the source and the target, which is what makes "train on
                NTU-Fi HAR, test on WiMANS" a real classification protocol rather than a label-space
                mismatch.

                A sample's label is a SET of activities, one per occupant, padded to
                preset["nn"]["num_obj_queries"] queries with the no-object class. That is exactly the
                format reduce_dataset() builds for the WiMANS activity task, so the same
                AMAR_WO_RVQ set-prediction head consumes both datasets: an NTU-Fi HAR capture holds
                one activity (its captures are single-person) and a WiMANS frame holds one per
                occupant.

                The WiMANS CSI keeps the repo's native (time, 3 TX, 3 RX, 30 subcarriers) flattening
                (270 features) so its side is untouched relative to the WiMANS baselines, and the
                NTU-Fi HAR subcarrier axis is resampled to match (see src/data/ntu_har.py).
"""
#
##

import os

import numpy as np

from configs.preset import preset
from src.data.load_data import load_data_y
from src.data.ntu_har import (NTU_HAR_FEATURES, load_ntu_har, ntu_har_class_names)
from src.utils import calculate_scores

#
##
## The activity vocabularies a cross-dataset run can be built on (see resolve_activity_map).
ACTIVITY_VOCABULARIES = ("shared", "walk_nothing", "union")
## WiMANS labels an occupant standing still as "nothing"; NTU-Fi HAR has no such capture, so the
## activity only ever appears on the target side.
WIMANS_IDLE_ACTIVITY = "nothing"


def wimans_activities():
    """
    [description]
    : WiMANS activity vocabulary, in the order preset["encoding"]["activity"] declares it.
    """
    return [var_name for var_name in preset["encoding"]["activity"] if var_name != "nan"]


def parse_activity_map(var_spec):
    """
    [description]
    : parse "--activity_map 'walk=walk,fall=lie_down'" into {ntu_class: wimans_activity}.
    """
    #
    ##
    var_map = {}
    for var_item in (var_spec or "").split(","):
        var_item = var_item.strip()
        if not var_item:
            continue
        var_ntu, var_sep, var_wimans = var_item.partition("=")
        if not var_sep:
            raise ValueError(f"--activity_map expects NTU=WiMANS pairs, got {var_item!r}")
        var_map[var_ntu.strip()] = var_wimans.strip()
    return var_map


def resolve_activity_map(var_map=None, var_ntu_classes=None, var_vocabulary=None):
    """
    [description]
    : the NTU-Fi HAR class -> shared-activity map a run uses, plus the resulting shared vocabulary.
    : var_map: explicit {ntu_class: wimans_activity}; it overrides var_vocabulary.
    : var_vocabulary: which activities the label space holds.
        "shared"       (default) the identity intersection of the two vocabularies: only activity
                       names both datasets use, so every target test frame is a class the source also
                       has. On NTU-Fi HAR x WiMANS that is {walk} alone, i.e. a walk-vs-no-object
                       detection task with no negatives.
        "walk_nothing" that intersection plus WiMANS's idle activity "nothing", so the target test is
                       walking against nothing. The source has no "nothing" captures, so the class is
                       target-only and only the few-shot student can learn it.
        "union"        every NTU-Fi HAR class and every WiMANS activity, mapped by name: "walk" is
                       shared, box/circle/clean exist on the source only and nothing/rotation/jump/
                       wave/lie_down/pick_up/sit_down/stand_up on the target only. Both datasets keep
                       all their labels and the transfer test spans the full vocabulary.
    : return: (var_map (dict), var_shared (sorted list of shared activity names))
    """
    #
    ##
    var_ntu_classes = (list(var_ntu_classes) if var_ntu_classes is not None
                       else ntu_har_class_names())
    var_wimans = wimans_activities()
    if var_map is not None:
        for var_ntu, var_wimans_name in var_map.items():
            if var_ntu not in var_ntu_classes:
                raise ValueError(f"activity map source {var_ntu!r} is not an NTU-Fi HAR class; "
                                 f"available {var_ntu_classes}")
            if var_wimans_name not in var_wimans:
                raise ValueError(f"activity map target {var_wimans_name!r} is not a WiMANS activity; "
                                 f"available {var_wimans}")
        var_shared = sorted(set(var_map.values()))
    else:
        var_vocabulary = var_vocabulary or "shared"
        var_common = {var_name: var_name for var_name in var_ntu_classes if var_name in var_wimans}
        if var_vocabulary == "shared":
            var_map, var_shared = var_common, sorted(set(var_common.values()))
        elif var_vocabulary == "walk_nothing":
            var_map = var_common
            var_shared = sorted(set(var_common.values()) | {WIMANS_IDLE_ACTIVITY})
        elif var_vocabulary == "union":
            ## every class keeps its own name; source-only classes (box/circle/clean) are simply
            ## never produced by a WiMANS frame and target-only activities never by an NTU-Fi HAR one
            var_map = {var_name: var_name for var_name in var_ntu_classes}
            var_shared = sorted(set(var_ntu_classes) | set(var_wimans))
        else:
            raise ValueError(f"var_vocabulary must be one of {ACTIVITY_VOCABULARIES}, "
                             f"got {var_vocabulary!r}")
    if not var_map:
        raise ValueError(
            f"NTU-Fi HAR and WiMANS share no activity name under vocabulary "
            f"{var_vocabulary!r}, so the cross-dataset run is undefined. Pass an explicit "
            f"--activity_map, e.g. --activity_map 'walk=walk,fall=lie_down', or --activities union.")
    if not var_shared:
        raise ValueError(f"vocabulary {var_vocabulary!r} is empty")
    return var_map, var_shared


def build_activity_set_labels(var_class_indices, var_num_queries, var_num_classes, var_seed=39):
    """
    [description]
    : one-hot activity-set labels for the AMAR_WO_RVQ set-prediction head. Each sample's activities
      fill the first queries and the remaining queries hold the no-object class (the last index), as
      reduce_dataset() does for the WiMANS activity task.
    : var_class_indices: list of per-sample index lists into the shared activity ordering (an empty
      list means no occupant)
    : return: (N, num_queries, num_classes) float32; the query order is a per-sample permutation drawn
      from a fixed seed, so a run is reproducible.
    """
    #
    ##
    var_rng = np.random.RandomState(var_seed)
    var_no_object = var_num_classes - 1
    var_y = np.zeros((len(var_class_indices), var_num_queries, var_num_classes), dtype=np.float32)
    var_y[:, :, var_no_object] = 1.0
    for var_i, var_classes in enumerate(var_class_indices):
        for var_slot, var_class in enumerate(var_classes[:var_num_queries]):
            var_y[var_i, var_slot] = 0.0
            var_y[var_i, var_slot, var_class] = 1.0
        var_y[var_i] = var_y[var_i][var_rng.permutation(var_num_queries)]
    return var_y


def standardize_windows(var_x, var_valid_lengths=None):
    """
    [description]
    : z-score every CSI window over its time x feature entries. This removes the per-device gain/AGC
      scale, which differs between NTU-Fi HAR and WiMANS and is the same cross-dataset alignment
      scripts/run_hwild_to_wimans.py applies.
    : var_x: (N, time, features) float32
    : var_valid_lengths: per-window number of real packets. WiMANS windows carry leading zero padding
      (the repo's convention, see load_data_x); passing the lengths restricts the statistics to the
      real tail, so the padding does not bias them.
    : return: same shape, float32
    """
    #
    ##
    var_x = np.asarray(var_x, dtype=np.float32)
    if var_valid_lengths is None:
        var_mu = var_x.mean(axis=(1, 2), keepdims=True)
        var_sd = var_x.std(axis=(1, 2), keepdims=True) + 1e-6
        return ((var_x - var_mu) / var_sd).astype(np.float32)
    #
    var_out = var_x.copy()
    for var_i, var_length in enumerate(var_valid_lengths):
        var_start = var_x.shape[1] - int(var_length)
        var_tail = var_x[var_i, var_start:]
        var_out[var_i, var_start:] = (var_tail - var_tail.mean()) / (var_tail.std() + 1e-6)
    return var_out


def harmonize_wimans_features(var_amp, var_features, var_length):
    """
    [description]
    : (time, 3 TX, 3 RX, 30 subcarriers) -> (var_length, var_features).
    : var_features: 270 keeps WiMANS's native flattening (3 x 3 x 30), the representation the repo's
      WiMANS runs of AMAR_WO_RVQ use; 90 averages the 3 TX antennas (axis 2 of (time, 3, 3, 30) ->
      (time, 3, 30)), the exact convention src/data/unified_rooms.py uses to match H-WILD.
    : The time axis keeps the repo's convention: shorter captures are zero-padded at the front, longer
      ones are an error (exactly as load_data_x does).
    : return: (var_length, var_features) float32, plus the number of real packets
    """
    #
    ##
    if var_features == 270:
        var_x = var_amp.reshape(var_amp.shape[0], -1)
    elif var_features == 90:
        var_x = var_amp.mean(axis=2).reshape(var_amp.shape[0], -1)
    else:
        raise ValueError(f"WiMANS supports var_features 270 (native) or 90 (TX-averaged), "
                         f"got {var_features}")
    var_num = var_x.shape[0]
    if var_num > var_length:
        raise ValueError(f"WiMANS capture has {var_num} packets, longer than length={var_length}")
    var_out = np.zeros((var_length, var_features), dtype=np.float32)
    var_out[var_length - var_num:] = var_x
    return var_out, var_num


def load_wimans_activity(var_activities, var_rooms=None, var_num_users=None, var_band=None,
                         var_length=None, var_features=None, var_max_samples=None,
                         var_standardize="window"):
    """
    [description]
    : load the WiMANS frames whose every occupant performs one of the shared activities, with the
      activity index of each occupant. Frames with an occupant doing an activity outside the shared
      vocabulary (and empty frames, which have no occupant at all) are dropped, so the label space is
      identical to the NTU-Fi HAR side.
    : var_activities: shared activity names, in the order that defines the class indices
    : var_rooms: rooms to include (default preset["data"]["environment"])
    : var_num_users: occupant-count filter (default ["1"], matching NTU-Fi HAR's single-person
      captures; widen it, e.g. "1,2,3", to also test set-size generalisation)
    : var_length: window length in packets (default preset["data"]["length"])
    : var_max_samples: cap per room with a fixed-seed random subset
    : return: dict {room: (X (N, length, features) float32, class_indices (list of index lists))}
    """
    #
    ##
    var_rooms = list(var_rooms) if var_rooms is not None else list(preset["data"]["environment"])
    var_band = var_band if var_band is not None else preset["data"]["wifi_band"]
    var_num_users = list(var_num_users) if var_num_users is not None else ["1"]
    var_length = var_length or preset["data"]["length"]
    var_features = var_features or preset.get("ntu_har", {}).get("features", NTU_HAR_FEATURES)
    var_index = {var_name: var_i for var_i, var_name in enumerate(var_activities)}
    var_cols = [f"user_{var_i}_activity" for var_i in range(1, 7)]
    #
    var_out = {}
    for var_room in var_rooms:
        var_pd = load_data_y(preset["path"]["data_y"], [var_room], var_band, var_num_users)
        var_labels, var_class_indices = [], []
        for _, var_row in var_pd.iterrows():
            ## every occupant's activity, ignoring the blank cells of absent users
            var_acts = [var_row[var_col] for var_col in var_cols
                        if isinstance(var_row[var_col], str) and var_row[var_col].strip()]
            ## a frame is only usable when *all* of its occupants do a shared activity: keeping it
            ## while dropping an out-of-vocabulary occupant would silently mislabel the frame
            if not var_acts or any(var_act not in var_index for var_act in var_acts):
                continue
            var_labels.append(var_row["label"])
            var_class_indices.append([var_index[var_act] for var_act in var_acts])
        if not var_labels:
            raise ValueError(
                f"no WiMANS frame in {var_room!r} has every occupant within the shared activities "
                f"{var_activities} (band {var_band}, occupants {var_num_users})")
        if var_max_samples is not None and len(var_labels) > var_max_samples:
            ## WiMANS is ordered single-user first, so a "first N" cut would drop whole groups;
            ## draw a fixed-seed random subset instead.
            var_keep = np.sort(np.random.RandomState(39).choice(len(var_labels), var_max_samples,
                                                                replace=False))
            var_labels = [var_labels[var_i] for var_i in var_keep]
            var_class_indices = [var_class_indices[var_i] for var_i in var_keep]
        #
        var_x = np.empty((len(var_labels), var_length, var_features), dtype=np.float32)
        var_lengths = np.zeros(len(var_labels), dtype=np.int64)
        for var_i, var_label in enumerate(var_labels):
            var_amp = np.load(os.path.join(preset["path"]["data_x"], var_label + ".npy")).astype(np.float32)
            var_x[var_i], var_lengths[var_i] = harmonize_wimans_features(var_amp, var_features,
                                                                        var_length)
        if var_standardize == "window":
            var_x = standardize_windows(var_x, var_lengths)
        elif var_standardize != "none":
            raise ValueError(f"var_standardize must be 'window' or 'none', got {var_standardize!r}")
        var_out[var_room] = (var_x, var_class_indices)
        print(f"[WiMANS] {var_room:12s} {var_x.shape} "
              f"({len(var_class_indices)} frames, activities {sorted(var_activities)})")
    return var_out


def split_target_pool(var_sizes, var_few_shot_ratio, var_valid_ratio=0.1, var_seed=39):
    """
    [description]
    : split the concatenated target pool into support / student-validation / query the way
      run_few_shot_localization does: one fixed-seed permutation, the support set is the first
      few_shot_ratio of it, the student's validation set the first valid_ratio of the remainder, and
      the query set the rest.
    : The pure-split and the few-shot runner both call this with the same arguments, so both are
      scored on exactly the same query frames and their numbers are directly comparable.
    : var_sizes: per-environment pool sizes, in concatenation order
    : return: (support_idx, valid_idx, query_idx) global index arrays
    """
    #
    ##
    var_num = int(np.sum(var_sizes))
    var_num_few = max(1, int(round(var_few_shot_ratio * var_num)))
    if var_num_few >= var_num - 1:
        raise ValueError(f"var_few_shot_ratio={var_few_shot_ratio} leaves no query data "
                         f"({var_num_few}/{var_num}). Lower the ratio or provide more frames.")
    var_perm = np.random.RandomState(var_seed).permutation(var_num)
    var_support_idx = np.sort(var_perm[:var_num_few])
    var_rest_idx = np.sort(var_perm[var_num_few:])
    var_num_valid = max(1, int(round(var_valid_ratio * len(var_rest_idx))))
    return var_support_idx, var_rest_idx[:var_num_valid], var_rest_idx[var_num_valid:]


def _env_slices(var_sizes):
    """
    [description]
    : (start, size) of every environment in the concatenated pool, in order.
    """
    var_slices, var_start = [], 0
    for var_size in var_sizes:
        var_slices.append((var_start, var_size))
        var_start += var_size
    return var_slices


def majority_activity(var_class_indices):
    """
    [description]
    : the most frequent shared activity index over a set of per-sample index lists (-1 if empty).
    """
    #
    ##
    var_counts = {}
    for var_sample in var_class_indices:
        for var_class in var_sample:
            var_counts[var_class] = var_counts.get(var_class, 0) + 1
    if not var_counts:
        return -1
    return max(sorted(var_counts), key=lambda var_class: var_counts[var_class])


def majority_query_metrics(var_majority, var_query_class_indices, var_num_classes):
    """
    [description]
    : metrics of the trivial predictor that always names one activity, computed with the same
      calculate_scores() the model reports use (activity counts summed over queries, no-object class
      dropped). On a shared-activity test set this is the reference a model has to beat: when the
      shared vocabulary holds a single activity, a model that always names it is already perfect, so
      a run that only matches this baseline has learned nothing.
    : return: dict of the same metric names calculate_scores produces
    """
    #
    ##
    var_true = np.zeros((len(var_query_class_indices), var_num_classes - 1), dtype=np.float32)
    for var_i, var_sample in enumerate(var_query_class_indices):
        for var_class in var_sample:
            var_true[var_i, var_class] += 1.0
    var_pred = np.zeros_like(var_true)
    if 0 <= var_majority < var_num_classes - 1:
        var_pred[:, var_majority] = 1.0
    return calculate_scores(var_true, var_pred)


def prepare_transfer_task(var_activity_map=None, var_ntu_root=None, var_ntu_splits=None,
                          var_features=None, var_wimans_rooms=None, var_wimans_users=None,
                          var_band=None, var_length=None, var_standardize="window",
                          var_max_ntu_files=None, var_max_wimans_samples=None,
                          var_few_shot_ratio=0.05, var_seed=39, var_vocabulary=None):
    """
    [description]
    : load both sides of the NTU-Fi HAR -> WiMANS shared-activity task and split the WiMANS pool into
      support / student-validation / query. Both runners call this, so they train on the same source
      data and are scored on the same target frames.
    : var_vocabulary: "shared" (default) / "walk_nothing" / "union", see resolve_activity_map; ignored
      when var_activity_map is given.
    : return: dict with
        var_map                    NTU-Fi HAR class -> shared activity
        var_shared                 shared activity names, the class ordering
        var_vocabulary             the vocabulary the class ordering came from
        var_num_classes            len(var_shared) + 1 (the no-object class)
        var_ntu_classes            NTU-Fi HAR classes kept
        train_sets_by_env          {"ntu_har": (X, y)}
        ntu_class_indices          per-NTU-sample shared activity indices
        pool_by_env                {room: (X, y)} the whole WiMANS pool
        support_idx/valid_idx/query_idx   global pool indices
        query_by_env               {room: (X, y)} the held-out query frames
        query_class_indices_by_env per-query shared activity indices
    """
    #
    ##
    var_map, var_shared = resolve_activity_map(
        var_activity_map,
        var_ntu_classes=ntu_har_class_names(var_ntu_root, var_ntu_splits),
        var_vocabulary=var_vocabulary)
    var_num_classes = len(var_shared) + 1
    var_num_queries = preset["nn"]["num_obj_queries"]
    var_features = var_features or preset.get("ntu_har", {}).get("features", NTU_HAR_FEATURES)
    #
    ## --------------------------------------- NTU-Fi HAR -------------------------------------------
    var_ntu_x, var_ntu_local_y, var_ntu_classes = load_ntu_har(
        var_root=var_ntu_root, var_splits=var_ntu_splits, var_classes=sorted(var_map),
        var_features=var_features, var_max_files=var_max_ntu_files)
    if var_standardize == "window":
        var_ntu_x = standardize_windows(var_ntu_x)
    elif var_standardize != "none":
        raise ValueError(f"var_standardize must be 'window' or 'none', got {var_standardize!r}")
    var_ntu_class_indices = [[var_shared.index(var_map[var_ntu_classes[int(var_local)]])]
                             for var_local in var_ntu_local_y]
    var_ntu_y = build_activity_set_labels(var_ntu_class_indices, var_num_queries, var_num_classes)
    print(f"[NTU-Fi HAR] {var_ntu_x.shape} | classes {var_ntu_classes} -> "
          f"{[var_map[c] for c in var_ntu_classes]}")
    #
    ## ----------------------------------------- WiMANS --------------------------------------------
    var_wimans = load_wimans_activity(var_shared, var_rooms=var_wimans_rooms,
                                      var_num_users=var_wimans_users, var_band=var_band,
                                      var_length=var_length, var_features=var_features,
                                      var_max_samples=var_max_wimans_samples,
                                      var_standardize=var_standardize)
    var_pool_by_env, var_sizes = {}, []
    for var_room, (var_x, _) in var_wimans.items():
        var_pool_by_env[var_room] = (var_x,
                                     build_activity_set_labels(var_wimans[var_room][1],
                                                               var_num_queries, var_num_classes))
        var_sizes.append(len(var_x))
    var_support_idx, var_valid_idx, var_query_idx = split_target_pool(var_sizes, var_few_shot_ratio,
                                                                     var_seed=var_seed)
    #
    ## the query frames per room, so the pure-split runner can be scored on the same frames the
    ## few-shot runner holds out
    var_query_by_env, var_query_class_indices_by_env = {}, {}
    for var_room, (var_start, var_size) in zip(var_pool_by_env, _env_slices(var_sizes)):
        var_mask = (var_query_idx >= var_start) & (var_query_idx < var_start + var_size)
        var_sel = var_query_idx[var_mask] - var_start
        var_x, var_y = var_pool_by_env[var_room]
        var_query_by_env[var_room] = (var_x[var_sel], var_y[var_sel])
        var_query_class_indices_by_env[var_room] = [var_wimans[var_room][1][int(var_i)]
                                                    for var_i in var_sel]
    #
    print(f"\nShared activities ({len(var_shared)}): {var_shared} | no-object class "
          f"{var_num_classes - 1} | num_classes {var_num_classes} | vocabulary "
          f"{var_vocabulary or ('explicit map' if var_activity_map is not None else 'shared')}")
    print(f"Activity map (NTU-Fi HAR -> class): {var_map}")
    print(f"WiMANS pool {int(np.sum(var_sizes))} frames {dict(zip(var_pool_by_env, var_sizes))} -> "
          f"support {len(var_support_idx)} ({var_few_shot_ratio:.2%}) | student validation "
          f"{len(var_valid_idx)} | query {len(var_query_idx)}")
    return {
        "var_map": var_map,
        "var_shared": var_shared,
        "var_vocabulary": var_vocabulary or ("map" if var_activity_map is not None else "shared"),
        "var_num_classes": var_num_classes,
        "var_ntu_classes": var_ntu_classes,
        "train_sets_by_env": {"ntu_har": (var_ntu_x, var_ntu_y)},
        "ntu_class_indices": var_ntu_class_indices,
        "pool_by_env": var_pool_by_env,
        "support_idx": var_support_idx,
        "valid_idx": var_valid_idx,
        "query_idx": var_query_idx,
        "query_by_env": var_query_by_env,
        "query_class_indices_by_env": var_query_class_indices_by_env,
    }
