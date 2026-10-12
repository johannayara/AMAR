"""
[file]          csi_preprocess.py
[description]   Pad-masked CSI amplitude preprocessing shared by every loader. load_data_x stores the
                amplitude front-zero-padded to preset["data"]["length"], so every statistic here is
                computed on the valid (non-pad) frames only. A statistic that includes the pad would
                encode the true sequence length, which is session/room dependent, i.e. it would leak
                the room identity instead of describing the signal.

                Every op rewrites the array in place, so a room that already occupies ~6 GB of float32
                gains no second copy. The ops are:
                  - per_room_zscore: standardize each feature with the room's own pad-masked mean/std.
                    Removes the room-specific gain/offset (AGC, NIC front-end) that otherwise lets the
                    network identify the room from a single scalar.
                  - static_removal: subtract each sample's pad-masked temporal mean per feature. The
                    time-average CSI is the static multipath, i.e. the room's fingerprint; removing it
                    leaves the motion-induced dynamic component that transfers across rooms.
                  - temporal_zscore: static_removal plus a per-feature division by the pad-masked
                    temporal std, normalizing the dynamic scale. This is the memory-safe realization
                    of "append mean/std as extra features": broadcasting them over 3000 frames would
                    triple the per-room array and OOM the node, and a channel that is constant over
                    time carries no temporal information the encoder could not already read from the
                    per-sample statistics it normalizes away.
"""
#
##
import os

import numpy as np

from configs.preset import preset

#
##
## The std floor keeps a near-static subcarrier from amplifying its own noise into a large feature.
_EPS = 1e-6
## Rows per accumulation block: bounds the transient (block, time, feature) temporaries.
_CHUNK = 64


def resolve_preprocess_config(var_config=None):
    """
    [description]
    : merge preset["data"]["preprocess"] with the AMAR_CSI_PREPROCESS environment override, so an
      ablation can be launched from a shell script without editing the preset.
    : var_config: optional dict, per-call override (takes precedence over the preset)
    : return: dict with the boolean keys static_removal, temporal_zscore, per_room_zscore
    """
    #
    ##
    var_resolved = dict(preset.get("data", {}).get("preprocess", {}))
    if var_config:
        var_resolved.update(var_config)
    #
    var_env = os.environ.get("AMAR_CSI_PREPROCESS")
    if var_env is not None:
        var_tokens = {var_token.strip().lower() for var_token in var_env.split(",") if var_token.strip()}
        if "none" in var_tokens:
            var_resolved.update(static_removal=False, temporal_zscore=False, per_room_zscore=False)
        else:
            var_resolved.update(static_removal="static" in var_tokens,
                                temporal_zscore="temporal" in var_tokens,
                                per_room_zscore="zscore" in var_tokens)
    #
    return {var_key: bool(var_resolved.get(var_key, False))
            for var_key in ("static_removal", "temporal_zscore", "per_room_zscore")}


def pad_mask(var_lengths, var_total_length):
    """
    [description]
    : boolean mask of the valid frames of front-padded sequences, True where the sample is real.
    : var_lengths: array (N,) of the original (unpadded) frame counts
    : var_total_length: padded length preset["data"]["length"]
    : return: bool array (N, var_total_length)
    """
    #
    var_lengths = np.asarray(var_lengths, dtype=np.int64)
    var_mask = np.zeros((len(var_lengths), var_total_length), dtype=bool)
    for var_idx, var_length in enumerate(var_lengths):
        var_length = int(min(max(var_length, 0), var_total_length))
        if var_length > 0:
            var_mask[var_idx, var_total_length - var_length:] = True
    return var_mask


def _masked_mean_std(var_x, var_mask):
    """
    [description]
    : per-sample per-feature mean and std over the valid frames, accumulated in blocks so no full
      (N, time, feature) temporary is ever materialized.
    : var_x: float32 array (N, time, feature)
    : var_mask: bool array (N, time)
    : return: (mean (N, feature) float64, std (N, feature) float64)
    """
    #
    var_num, var_time, var_feat = var_x.shape
    var_mask_float = var_mask.astype(np.float32)
    var_counts = np.maximum(var_mask_float.sum(axis=1), 1.0)
    var_mean = np.empty((var_num, var_feat), dtype=np.float64)
    var_std = np.empty((var_num, var_feat), dtype=np.float64)
    #
    for var_start in range(0, var_num, _CHUNK):
        var_stop = min(var_start + _CHUNK, var_num)
        var_x_block = var_x[var_start:var_stop]
        var_mask_block = var_mask_float[var_start:var_stop, :, None]
        var_count_block = var_counts[var_start:var_stop, None]
        var_mean[var_start:var_stop] = (var_x_block * var_mask_block).sum(axis=1) / var_count_block
        var_center = var_x_block - var_mean[var_start:var_stop][:, None, :]
        var_std[var_start:var_stop] = np.sqrt(
            np.maximum((var_center * var_center * var_mask_block).sum(axis=1) / var_count_block, 0.0))
    #
    return var_mean, var_std


def _combine_room_stats(var_mean, var_std, var_counts):
    """
    [description]
    : pool per-sample statistics into one per-feature room statistic, weighting each sample by its
      valid-frame count so padded samples do not dominate.
    : return: (room_mean (feature,), room_std (feature,)) float64
    """
    #
    var_total = max(float(np.sum(var_counts)), 1.0)
    var_room_mean = (var_mean * var_counts[:, None]).sum(axis=0) / var_total
    var_ex2 = ((var_std ** 2 + var_mean ** 2) * var_counts[:, None]).sum(axis=0) / var_total
    var_room_std = np.sqrt(np.maximum(var_ex2 - var_room_mean ** 2, 0.0))
    return var_room_mean, var_room_std


def preprocess_csi(var_x, var_lengths, var_config=None):
    """
    [description]
    : apply the configured pad-masked preprocessing in place. With every op disabled the input is
      returned untouched (and keeps its original trailing shape); with any op enabled the trailing
      axes are flattened to (N, time, feature), which is the shape the models consume anyway.
    : var_x: float32 array (N, time, ...) as returned by load_data_x
    : var_lengths: array (N,) of the original frame counts
    : var_config: optional override, see resolve_preprocess_config
    : return: float32 array, preprocessed
    """
    #
    var_cfg = resolve_preprocess_config(var_config)
    if not any(var_cfg.values()) or var_x.size == 0:
        return var_x
    #
    var_num, var_time = var_x.shape[0], var_x.shape[1]
    var_flat = np.ascontiguousarray(var_x.reshape(var_num, var_time, -1), dtype=np.float32)
    var_mask = pad_mask(var_lengths, var_time)
    var_mask_float = var_mask.astype(np.float32)
    #
    ## -------------------------------------- per-room z-score -------------------------------------- ##
    #
    if var_cfg["per_room_zscore"]:
        var_mean, var_std = _masked_mean_std(var_flat, var_mask)
        var_room_mean, var_room_std = _combine_room_stats(
            var_mean, var_std, np.maximum(var_mask_float.sum(axis=1), 1.0))
        var_scale = np.maximum(var_room_std, _EPS).astype(np.float32)
        for var_start in range(0, var_num, _CHUNK):
            var_stop = min(var_start + _CHUNK, var_num)
            var_block_mask = var_mask[var_start:var_stop, :, None]
            var_centered = (var_flat[var_start:var_stop] - var_room_mean.astype(np.float32)) / var_scale
            var_flat[var_start:var_stop] = np.where(var_block_mask, var_centered, 0.0)
    #
    ## --------------------------- per-sample static removal / z-score ------------------------------ ##
    #
    if var_cfg["temporal_zscore"] or var_cfg["static_removal"]:
        var_mean, var_std = _masked_mean_std(var_flat, var_mask)
        var_scale = np.maximum(var_std, _EPS).astype(np.float32) if var_cfg["temporal_zscore"] else None
        for var_start in range(0, var_num, _CHUNK):
            var_stop = min(var_start + _CHUNK, var_num)
            var_block_mask = var_mask[var_start:var_stop, :, None]
            var_centered = var_flat[var_start:var_stop] - var_mean[var_start:var_stop].astype(np.float32)[:, None, :]
            if var_scale is not None:
                var_centered = var_centered / var_scale[var_start:var_stop][:, None, :]
            var_flat[var_start:var_stop] = np.where(var_block_mask, var_centered, 0.0)
    #
    return var_flat
