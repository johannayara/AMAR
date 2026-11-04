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
from src.data.load_data import (load_data_y, harmonize_wimans_antennas, wimans_room_names,
                                wimans_room_source)
from src.data.ntu_har import (NTU_HAR_FEATURES, load_ntu_har, ntu_har_class_names)
from src.utils import calculate_scores

#
##
## The activity vocabularies a cross-dataset run can be built on (see resolve_vocabulary).
ACTIVITY_VOCABULARIES = ("shared", "walk_nothing", "union")
## WiMANS labels an occupant standing still as "nothing"; NTU-Fi HAR has no such capture, so the
## activity only ever appears on the target side.
WIMANS_IDLE_ACTIVITY = "nothing"
## Label aliases for activity names a WiMANS-format annotation uses that the preset vocabulary does
## not name directly. The simulated few-shot dataset
## (dataset/synth_wimans/fewshot-groups-wimans) labels a sitting occupant "sit"; WiMANS's own
## vocabulary splits that into "sit_down"/"stand_up" and the simulator's action behind it is "sitting"
## (never standing), so it maps to sit_down. Without the alias load_wimans_activity's vocabulary check
## drops every frame holding a "sit" occupant -- 1478 of the simulated set's 2715 frames.
WIMANS_ACTIVITY_ALIASES = {"sit": "sit_down"}


def wimans_activities():
    """
    [description]
    : WiMANS activity vocabulary, in the order preset["encoding"]["activity"] declares it.
    """
    return [var_name for var_name in preset["encoding"]["activity"] if var_name != "nan"]


def resolve_vocabulary(var_vocabulary=None, var_ntu_classes=None):
    """
    [description]
    : the label space a cross-dataset run uses. Activities are matched by name -- a class keeps its
      own name on both sides -- so the vocabulary fixes which NTU-Fi HAR classes the source provides
      and which activity names the class ordering holds.
    : var_vocabulary: which activities the label space holds.
        "shared"       (default) the identity intersection of the two vocabularies: only activity
                       names both datasets use, so every target test frame is a class the source also
                       has. On NTU-Fi HAR x WiMANS that is {walk} alone, i.e. a walk-vs-no-object
                       detection task with no negatives.
        "walk_nothing" that intersection plus WiMANS's idle activity "nothing", so the target test is
                       walking against nothing. The source has no "nothing" captures, so the class is
                       target-only and only the few-shot student can learn it.
        "union"        every NTU-Fi HAR class and every WiMANS activity: "walk" is shared,
                       box/circle/clean exist on the source only and nothing/rotation/jump/wave/
                       lie_down/pick_up/sit_down/stand_up on the target only. Both datasets keep all
                       their labels and the transfer test spans the full vocabulary.
    : return: (var_ntu_kept (sorted NTU-Fi HAR class names the source loads), var_shared (sorted list
      of activity names, the class ordering))
    """
    #
    ##
    var_ntu_classes = (list(var_ntu_classes) if var_ntu_classes is not None
                       else ntu_har_class_names())
    var_wimans = wimans_activities()
    var_vocabulary = var_vocabulary or "shared"
    if var_vocabulary == "shared":
        var_ntu_kept = [var_name for var_name in var_ntu_classes if var_name in var_wimans]
        var_shared = sorted(var_ntu_kept)
    elif var_vocabulary == "walk_nothing":
        var_ntu_kept = [var_name for var_name in var_ntu_classes if var_name in var_wimans]
        var_shared = sorted(set(var_ntu_kept) | {WIMANS_IDLE_ACTIVITY})
    elif var_vocabulary == "union":
        var_ntu_kept = list(var_ntu_classes)
        var_shared = sorted(set(var_ntu_classes) | set(var_wimans))
    else:
        raise ValueError(f"var_vocabulary must be one of {ACTIVITY_VOCABULARIES}, "
                         f"got {var_vocabulary!r}")
    if not var_ntu_kept:
        raise ValueError(
            f"NTU-Fi HAR and WiMANS share no activity name under vocabulary {var_vocabulary!r}, so "
            f"the cross-dataset run is undefined. Use --activities union.")
    return sorted(var_ntu_kept), var_shared


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


def standardize_windows(var_x, var_valid_lengths=None, var_in_place=False):
    """
    [description]
    : z-score every CSI window over its time x feature entries. This removes the per-device gain/AGC
      scale, which differs between NTU-Fi HAR and WiMANS and is the same cross-dataset alignment
      scripts/run_hwild_to_wimans.py applies.
    : var_x: (N, time, features) float32
    : var_valid_lengths: per-window number of real packets. WiMANS windows carry leading zero padding
      (the repo's convention, see load_data_x); passing the lengths restricts the statistics to the
      real tail, so the padding does not bias them.
    : var_in_place: write into var_x instead of returning a copy, for callers that own the array. The
      WiMANS pool is (N, 3000, 270) float32 -- several GB on the full vocabulary -- so the copy is
      worth avoiding there; the numbers are identical either way.
    : return: same shape, float32
    """
    #
    ##
    var_x = np.asarray(var_x, dtype=np.float32)
    if var_valid_lengths is None:
        var_mu = var_x.mean(axis=(1, 2), keepdims=True)
        var_sd = var_x.std(axis=(1, 2), keepdims=True) + 1e-6
        if var_in_place:
            var_x -= var_mu
            var_x /= var_sd
            return var_x
        return ((var_x - var_mu) / var_sd).astype(np.float32)
    #
    var_out = var_x if var_in_place else var_x.copy()
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
    : A single-link capture (time, 1, 1, 30) is first broadcast to the 3 x 3 grid by
      harmonize_wimans_antennas, so the simulated few-shot dataset loads through the same path.
    : The time axis keeps the repo's convention: shorter captures are zero-padded at the front, longer
      ones are an error (exactly as load_data_x does).
    : return: (var_length, var_features) float32, plus the number of real packets
    """
    #
    ##
    var_amp = harmonize_wimans_antennas(var_amp)
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


def select_wimans_frames(var_activities, var_room, var_num_users, var_band, var_max_samples=None):
    """
    [description]
    : select one room's WiMANS frames whose every occupant performs one of the shared activities, with
      the activity index of each occupant. Frames with an occupant doing an activity outside the shared
      vocabulary (and empty frames, which have no occupant at all) are dropped, so the label space is
      identical to the NTU-Fi HAR side.
    : Annotation only -- no CSI is read -- so a run can resolve the whole pool's frame selection and
      still hold none of its frames (see WimansFrameSource).
    : var_activities: shared activity names, in the order that defines the class indices
    : var_room: the room key to select from (see wimans_room_names)
    : var_num_users: occupant-count filter (default ["1"], matching NTU-Fi HAR's single-person
      captures; widen it, e.g. "1,2,3", to also test set-size generalisation)
    : var_max_samples: cap with a fixed-seed random subset
    : var_band: annotation band filter; None (default) uses the room's own source band, so real 5 GHz
      rooms and simulated 2.4 GHz rooms can be selected in the same run
    : return: (var_labels, var_class_indices, var_data_x) -- the selected captures, their per-frame
      activity index lists, and the CSI directory they live in
    """
    #
    ##
    var_num_users = list(var_num_users) if var_num_users is not None else ["1"]
    var_index = {var_name: var_i for var_i, var_name in enumerate(var_activities)}
    var_cols = [f"user_{var_i}_activity" for var_i in range(1, 7)]
    ## each room key resolves to its own source: the real rooms read preset["path"], the simulated
    ## keys read preset["data"]["synth_wimans"], and both use the same base room name
    var_base, var_data_x, var_data_y, var_source_band = wimans_room_source(var_room)
    var_room_band = list(var_band) if var_band is not None else var_source_band
    var_pd = load_data_y(var_data_y, [var_base], var_room_band, var_num_users)
    var_labels, var_class_indices = [], []
    for _, var_row in var_pd.iterrows():
        ## every occupant's activity, ignoring the blank cells of absent users
        var_acts = [WIMANS_ACTIVITY_ALIASES.get(var_row[var_col], var_row[var_col])
                    for var_col in var_cols
                    if isinstance(var_row[var_col], str) and var_row[var_col].strip()]
        ## a frame is only usable when *all* of its occupants do a shared activity: keeping it while
        ## dropping an out-of-vocabulary occupant would silently mislabel the frame
        if not var_acts or any(var_act not in var_index for var_act in var_acts):
            continue
        var_labels.append(var_row["label"])
        var_class_indices.append([var_index[var_act] for var_act in var_acts])
    if not var_labels:
        raise ValueError(
            f"no WiMANS frame in {var_room!r} has every occupant within the shared activities "
            f"{var_activities} (band {var_room_band}, occupants {var_num_users})")
    if var_max_samples is not None and len(var_labels) > var_max_samples:
        ## WiMANS is ordered single-user first, so a "first N" cut would drop whole groups; draw a
        ## fixed-seed random subset instead.
        var_keep = np.sort(np.random.RandomState(39).choice(len(var_labels), var_max_samples,
                                                            replace=False))
        var_labels = [var_labels[var_i] for var_i in var_keep]
        var_class_indices = [var_class_indices[var_i] for var_i in var_keep]
    return var_labels, var_class_indices, var_data_x


def load_wimans_frames(var_labels, var_data_x, var_length, var_features, var_standardize):
    """
    [description]
    : read and standardize the named WiMANS captures: (len(var_labels), var_length, var_features)
      float32. The only place WiMANS CSI is read, so the eager pool and the lazy WimansFrameSource get
      the same padding (load_data_x's leading zero padding) and the same window z-score.
    """
    #
    ##
    var_x = np.empty((len(var_labels), var_length, var_features), dtype=np.float32)
    var_lengths = np.zeros(len(var_labels), dtype=np.int64)
    for var_i, var_label in enumerate(var_labels):
        var_amp = np.load(os.path.join(var_data_x, var_label + ".npy")).astype(np.float32)
        var_x[var_i], var_lengths[var_i] = harmonize_wimans_features(var_amp, var_features,
                                                                    var_length)
    if var_standardize == "window":
        ## in place: the pool is freshly allocated here and is the largest array in the run
        var_x = standardize_windows(var_x, var_lengths, var_in_place=True)
    elif var_standardize != "none":
        raise ValueError(f"var_standardize must be 'window' or 'none', got {var_standardize!r}")
    return var_x


class WimansFrameSource:
    """
    [description]
    : lazy array-like view of one WiMANS room's selected frames. Indexing it reads only the requested
      captures, so a run that walks the pool (support/validation gather, query evaluation, clustering)
      never holds more than the frames it is using at that moment. The pool is (frames, 3000, 270)
      float32 -- ~3 MiB per frame -- so the union pool at --wimans_users 1,2,3 is ~20 GiB, which is
      what pushes the 32 GB job over once the support/validation copies and the source data sit beside
      it. Supports len(), .shape, .dtype and int / slice / integer-array indexing, so it drops into
      the places that expect the (N, length, features) array.
    : var_labels: this room's selected captures, in pool order (see select_wimans_frames)
    : var_data_x: the room's CSI directory
    : var_length / var_features / var_standardize: padding / feature / standardization settings
    """
    #
    ##
    def __init__(self, var_labels, var_data_x, var_length, var_features, var_standardize):
        self.labels = list(var_labels)
        self.data_x = var_data_x
        self.length = var_length
        self.features = var_features
        self.standardize = var_standardize

    def __len__(self):
        return len(self.labels)

    @property
    def shape(self):
        return (len(self.labels), self.length, self.features)

    @property
    def dtype(self):
        return np.dtype(np.float32)

    def __getitem__(self, var_index):
        var_ids = np.arange(len(self.labels))[var_index]
        var_scalar = np.ndim(var_ids) == 0
        var_ids = np.atleast_1d(var_ids).astype(np.int64)
        var_x = load_wimans_frames([self.labels[int(var_i)] for var_i in var_ids], self.data_x,
                                   self.length, self.features, self.standardize)
        return var_x[0] if var_scalar else var_x


def load_wimans_activity(var_activities, var_rooms=None, var_num_users=None, var_band=None,
                         var_length=None, var_features=None, var_max_samples=None,
                         var_standardize="window"):
    """
    [description]
    : load the WiMANS frames whose every occupant performs one of the shared activities, with the
      activity index of each occupant. Frames with an occupant doing an activity outside the shared
      vocabulary (and empty frames, which have no occupant at all) are dropped, so the label space is
      identical to the NTU-Fi HAR side.
    : The eager counterpart of WimansFrameSource: every room's frames are materialized up front. Use
      the lazy source (prepare_transfer_task's var_stream_pool) when the pool is too large to hold.
    : var_activities: shared activity names, in the order that defines the class indices
    : var_rooms: rooms to include (default every key of wimans_room_names(): the real WiMANS
      environments, plus "<name><suffix>" for the simulated source when it is enabled)
    : var_num_users: occupant-count filter (default ["1"], matching NTU-Fi HAR's single-person
      captures; widen it, e.g. "1,2,3", to also test set-size generalisation)
    : var_length: window length in packets (default preset["data"]["length"])
    : var_max_samples: cap per room with a fixed-seed random subset
    : var_band: annotation band filter for every room; None (default) uses each room's own source
      band, so real 5 GHz rooms and simulated 2.4 GHz rooms can be loaded in the same run
    : return: dict {room: (X (N, length, features) float32, class_indices (list of index lists))}
    """
    #
    ##
    var_rooms = (list(var_rooms) if var_rooms is not None
                 else wimans_room_names())
    var_length = var_length or preset["data"]["length"]
    var_features = var_features or preset.get("ntu_har", {}).get("features", NTU_HAR_FEATURES)
    var_out = {}
    for var_room in var_rooms:
        var_labels, var_class_indices, var_data_x = select_wimans_frames(
            var_activities, var_room, var_num_users, var_band, var_max_samples)
        var_x = load_wimans_frames(var_labels, var_data_x, var_length, var_features, var_standardize)
        var_out[var_room] = (var_x, var_class_indices)
        print(f"[WiMANS] {var_room:12s} {var_x.shape} "
              f"({len(var_class_indices)} frames, activities {sorted(var_activities)})")
    return var_out


def split_target_pool(var_sizes, var_few_shot_ratio, var_valid_ratio=0.1, var_seed=39,
                      var_support_sizes=None):
    """
    [description]
    : split the concatenated target pool into support / student-validation / query: one fixed-seed
      permutation, the support set is the first few_shot_ratio of it, the student's validation set the
      first valid_ratio of the remainder, and the query set the rest. Each split is returned sorted,
      but only for indexing convenience -- the validation slice is cut from the permutation order, not
      from the sorted indices, so it spans every environment and activity rather than sitting at the
      start of the pool.
    : The pure-split and the few-shot runner both call this with the same arguments, so both are
      scored on exactly the same query frames and their numbers are directly comparable.
    : var_sizes: per-environment pool sizes, in concatenation order
    : var_support_sizes: per-environment sizes eligible to supply the support and validation frames
      (default var_sizes, i.e. the whole pool). A pool that concatenates real WiMANS with a second
      source passes the real sizes here, so the student is fine-tuned and checkpoint-selected on the
      real rooms only while the query set still holds every frame of both sources -- which is what
      "the simulated rooms are in the test set, not the training set" means.
    : return: (support_idx, valid_idx, query_idx) global index arrays
    """
    #
    ##
    var_num = int(np.sum(var_sizes))
    var_sizes = list(var_sizes)
    var_support_sizes = list(var_support_sizes) if var_support_sizes is not None else list(var_sizes)
    ## the rows a support/validation frame may be drawn from, ascending
    var_eligible = np.concatenate([np.arange(var_start, var_start + var_size)
                                   for (var_start, _), var_size in zip(_env_slices(var_sizes),
                                                                       var_support_sizes)])
    var_num_few = max(1, int(round(var_few_shot_ratio * len(var_eligible))))
    if var_num_few >= var_num - 1:
        raise ValueError(f"var_few_shot_ratio={var_few_shot_ratio} leaves no query data "
                         f"({var_num_few}/{var_num}). Lower the ratio or provide more frames.")
    var_perm = np.random.RandomState(var_seed).permutation(var_eligible)
    var_support_idx = np.sort(var_perm[:var_num_few])
    ## The remainder keeps the permutation order while the validation slice is cut off it: sorting it
    ## first would make "the first valid_ratio of the remainder" mean "the remainder's lowest pool
    ## indices", and the pool is concatenated room by room in annotation order -- WiMANS groups a
    ## room's frames by activity -- so the student's validation set would be one room's first
    ## activities rather than a slice of every room. That would select the checkpoint on one room and
    ## leave the per-room query counts unequal (378/532/534 instead of ~481 each on the union pool).
    var_rest_idx = var_perm[var_num_few:]
    var_num_valid = max(1, int(round(var_valid_ratio * len(var_rest_idx))))
    var_valid_idx = np.sort(var_rest_idx[:var_num_valid])
    ## the query set is every pool row that is neither support nor validation, so it holds the whole
    ## of every source, the ones the student was fine-tuned on excepted
    var_taken = np.zeros(var_num, dtype=bool)
    var_taken[var_support_idx] = True
    var_taken[var_valid_idx] = True
    return var_support_idx, var_valid_idx, np.where(~var_taken)[0]


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


def prepare_transfer_task(var_ntu_root=None, var_ntu_splits=None,
                          var_features=None, var_wimans_rooms=None, var_wimans_users=None,
                          var_band=None, var_length=None, var_standardize="window",
                          var_max_ntu_files=None, var_max_wimans_samples=None,
                          var_few_shot_ratio=0.05, var_seed=39, var_vocabulary=None,
                          var_with_pool=True, var_with_query=True, var_stream_pool=False):
    """
    [description]
    : load both sides of the NTU-Fi HAR -> WiMANS shared-activity task and split the WiMANS pool into
      support / student-validation / query. Both runners call this, so they train on the same source
      data and are scored on the same target frames.
    : var_vocabulary: "shared" (default) / "walk_nothing" / "union", see resolve_vocabulary.
    : var_with_pool / var_with_query: whether to hand back the per-room WiMANS pool and the per-room
      query frames. The pool is (frames, 3000, 270) float32 -- 5.8 GB over the full union vocabulary
      (1782 frames) -- and the query frames are a second copy of 80% of it, so a runner that needs
      only one of the two asks for only that one and the other is not materialized. The few-shot
      runner indexes the pool with support_idx/valid_idx/query_idx; the pure-split runner is scored on
      the query frames. Only the returned dict is affected: the split indices and the per-query
      activity labels are always built.
    : var_stream_pool: hand back the pool as a lazy per-room source (WimansFrameSource) instead of
      materializing every room's frames. The selection (labels + activity indices) is still resolved
      for the whole pool, but a room's CSI is read only when the pool is indexed, so a run that walks
      it (support/validation gather, query evaluation, clustering) never holds more than the frames it
      is using. Required once the pool outgrows the job: the union pool at --wimans_users 1,2,3 is
      ~20 GiB of frames, which does not fit a 32 GB job beside the support/validation copies and the
      source data. The frame selection, split indices and every metric are identical to the eager
      pool, so streamed and eager runs stay comparable. Needs var_with_pool; var_with_query
      materializes the query frames off the source, defeating the point.
    : return: dict with
        var_shared                 shared activity names, the class ordering
        var_vocabulary             the vocabulary the class ordering came from
        var_num_classes            len(var_shared) + 1 (the no-object class)
        var_ntu_classes            NTU-Fi HAR classes kept
        train_sets_by_env          {"ntu_har": (X, y)}
        ntu_class_indices          per-NTU-sample shared activity indices
        pool_by_env                {room: (X, y)} the whole WiMANS pool, {} unless var_with_pool;
                                   X is a WimansFrameSource when var_stream_pool, else an array
        support_idx/valid_idx/query_idx   global pool indices
        query_by_env               {room: (X, y)} the held-out query frames, {} unless var_with_query
        query_class_indices_by_env per-query shared activity indices
    """
    #
    ##
    var_ntu_kept, var_shared = resolve_vocabulary(
        var_vocabulary, var_ntu_classes=ntu_har_class_names(var_ntu_root, var_ntu_splits))
    var_vocabulary = var_vocabulary or "shared"
    var_num_classes = len(var_shared) + 1
    var_num_queries = preset["nn"]["num_obj_queries"]
    var_features = var_features or preset.get("ntu_har", {}).get("features", NTU_HAR_FEATURES)
    #
    ## --------------------------------------- NTU-Fi HAR -------------------------------------------
    var_ntu_x, var_ntu_local_y, var_ntu_classes = load_ntu_har(
        var_root=var_ntu_root, var_splits=var_ntu_splits, var_classes=var_ntu_kept,
        var_features=var_features, var_max_files=var_max_ntu_files)
    if var_standardize == "window":
        var_ntu_x = standardize_windows(var_ntu_x, var_in_place=True)
    elif var_standardize != "none":
        raise ValueError(f"var_standardize must be 'window' or 'none', got {var_standardize!r}")
    ## a class keeps its own name, so the local NTU-Fi HAR index maps straight into the shared ordering
    var_ntu_class_indices = [[var_shared.index(var_ntu_classes[int(var_local)])]
                             for var_local in var_ntu_local_y]
    var_ntu_y = build_activity_set_labels(var_ntu_class_indices, var_num_queries, var_num_classes)
    print(f"[NTU-Fi HAR] {var_ntu_x.shape} | classes {var_ntu_classes}")
    #
    ## ----------------------------------------- WiMANS --------------------------------------------
    ## var_stream_pool resolves only the frame selection (annotation) per room and defers the CSI to
    ## WimansFrameSource; otherwise every room is materialized up front.
    var_pool_by_env, var_class_indices_by_env = {}, {}
    if var_stream_pool:
        var_length = var_length or preset["data"]["length"]
        var_room_names = (list(var_wimans_rooms) if var_wimans_rooms is not None
                          else wimans_room_names())
        for var_room in var_room_names:
            var_labels, var_class_indices, var_data_x = select_wimans_frames(
                var_shared, var_room, var_wimans_users, var_band, var_max_wimans_samples)
            var_pool_by_env[var_room] = (
                WimansFrameSource(var_labels, var_data_x, var_length, var_features, var_standardize),
                build_activity_set_labels(var_class_indices, var_num_queries, var_num_classes))
            var_class_indices_by_env[var_room] = var_class_indices
            print(f"[WiMANS] {var_room:12s} {tuple(var_pool_by_env[var_room][0].shape)} "
                  f"({len(var_class_indices)} frames, activities {sorted(var_shared)})")
    else:
        var_wimans = load_wimans_activity(var_shared, var_rooms=var_wimans_rooms,
                                          var_num_users=var_wimans_users, var_band=var_band,
                                          var_length=var_length, var_features=var_features,
                                          var_max_samples=var_max_wimans_samples,
                                          var_standardize=var_standardize)
        var_room_names = list(var_wimans)
        for var_room in var_room_names:
            var_class_indices_by_env[var_room] = var_wimans[var_room][1]
            var_pool_by_env[var_room] = (
                var_wimans[var_room][0],
                build_activity_set_labels(var_wimans[var_room][1], var_num_queries, var_num_classes))
    ## the set labels are one row per pool frame, so their length is the room's frame count
    var_sizes = [len(var_pool_by_env[var_room][1]) for var_room in var_room_names]
    ## The support and validation frames come from the real WiMANS rooms only. A second source (the
    ## simulated set, enabled through preset["data"]["synth_wimans"]) contributes query frames but
    ## must not be fine-tuned on or select the checkpoint, which is what "in the test set" means.
    var_real_envs = set(preset["data"]["environment"])
    var_support_sizes = [var_size for var_room, var_size in zip(var_room_names, var_sizes)
                         if var_room in var_real_envs]
    if len(var_support_sizes) == len(var_sizes):
        var_support_sizes = None
    var_support_idx, var_valid_idx, var_query_idx = split_target_pool(var_sizes, var_few_shot_ratio,
                                                                     var_seed=var_seed,
                                                                     var_support_sizes=var_support_sizes)
    #
    ## the query frames per room, so the pure-split runner can be scored on the same frames the
    ## few-shot runner holds out; the per-query activity labels are cheap and always built
    var_query_by_env, var_query_class_indices_by_env = {}, {}
    for var_room, (var_start, var_size) in zip(var_room_names, _env_slices(var_sizes)):
        var_mask = (var_query_idx >= var_start) & (var_query_idx < var_start + var_size)
        var_sel = var_query_idx[var_mask] - var_start
        if var_with_query:
            var_x, var_y = var_pool_by_env[var_room]
            var_query_by_env[var_room] = (var_x[var_sel], var_y[var_sel])
        var_query_class_indices_by_env[var_room] = [var_class_indices_by_env[var_room][int(var_i)]
                                                    for var_i in var_sel]
    if not var_with_pool:
        ## the pool's only other use is run_cross_domain's concatenation, which the pure-split runner
        ## does off the query frames; releasing it here keeps the two views from being resident at once
        var_pool_by_env = {}
    #
    print(f"\nShared activities ({len(var_shared)}): {var_shared} | no-object class "
          f"{var_num_classes - 1} | num_classes {var_num_classes} | vocabulary {var_vocabulary}")
    print(f"WiMANS pool {int(np.sum(var_sizes))} frames {dict(zip(var_room_names, var_sizes))} -> "
          f"support {len(var_support_idx)} ({var_few_shot_ratio:.2%}) | student validation "
          f"{len(var_valid_idx)} | query {len(var_query_idx)}")
    return {
        "var_shared": var_shared,
        "var_vocabulary": var_vocabulary,
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
