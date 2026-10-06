"""
[file]          unified_rooms.py
[description]   Unified room registry over the WiMANS and H-WILD datasets, for room-agnostic
                leave-one-room-out localization on the pooled room set.

                Every room is expressed in its own [0, 1]^2 frame defined by the room's physical
                extent, so a model trained on rooms from either dataset predicts normalized
                coordinates that denormalize to meters in any room:
                  - H-WILD: the coordinate grid of obtain_parameters.m (src/data/hwild.py).
                  - WiMANS: the room rectangle (510 x 1030 cm, origin at the transmitter), with each
                    user's position taken from their occupied discrete location.

                Labels are SETS of positions, not a single point: a WiMANS sample keeps all of its
                occupied locations (up to MAX_USERS, one per user), and an H-WILD sample is a
                one-element set (the person's UWB position). This preserves the multi-user structure
                the density-map formulation is built on. Positions are returned padded to MAX_USERS
                with a 0/1 mask marking the valid entries.

                Feature vectors are made compatible: WiMANS CSI is (time, 3, 3, 30) = 270 values, so
                the 3 TX antennas are averaged to the 3 x 30 = 90 layout H-WILD uses. WiMANS samples
                are resampled from their native length to the shared window length.
"""
#
##

import os

import numpy as np
from scipy.signal import resample_poly

from configs.preset import preset
from src.data.load_data import load_data_y
from src.data.hwild import (HWILD_ROOMS, room_names as hwild_room_names, load_hwild_room,
                            room_span as hwild_room_span)

#
##
WIMANS_ROOMS = ["empty_room", "meeting_room", "classroom"]
## WiMANS rooms are 510 x 1030 cm (preset["layouts"] comment); layout coordinates are in units of
## 1000 cm, i.e. 1 unit = 10 m, with the origin at the transmitter.
WIMANS_EXTENT_M = (5.10, 10.30)
WIMANS_BOUNDS = (0.0, WIMANS_EXTENT_M[0], 0.0, WIMANS_EXTENT_M[1])
WIMANS_FEATURES = 90  # 3 antennas x 30 subcarriers after averaging the 3 TX antennas
MAX_USERS = 6         # WiMANS labels up to 6 users; every room's label set is padded to this size


def unified_room_names():
    """
    [description]
    : every room of both datasets, WiMANS first.
    """
    return WIMANS_ROOMS + hwild_room_names()


def dataset_of(var_room):
    """
    [description]
    : which dataset a room belongs to ("wimans" or "hwild").
    """
    return "wimans" if var_room in WIMANS_ROOMS else "hwild"


def room_bounds(var_room):
    """
    [description]
    : (x_min, x_max, y_min, y_max) of a room's frame in meters.
    """
    if var_room in WIMANS_ROOMS:
        return WIMANS_BOUNDS
    var_room_meta = HWILD_ROOMS[var_room]
    return (var_room_meta["x_range"][0], var_room_meta["x_range"][1],
            var_room_meta["y_range"][0], var_room_meta["y_range"][1])


def room_span(var_room):
    """
    [description]
    : (x_span, y_span) of a room's frame in meters.
    """
    var_bounds = room_bounds(var_room)
    return (var_bounds[1] - var_bounds[0], var_bounds[3] - var_bounds[2])


def normalize_xy_meters(var_xy_m, var_bounds):
    """
    [description]
    : map raw positions in meters into the room's [0, 1]^2 frame.
    """
    var_x0, var_x1, var_y0, var_y1 = var_bounds
    var_out = np.array(var_xy_m, dtype=np.float32, copy=True)
    var_out[..., 0] = (var_out[..., 0] - var_x0) / (var_x1 - var_x0)
    var_out[..., 1] = (var_out[..., 1] - var_y0) / (var_y1 - var_y0)
    return var_out


def _resample_time(var_x, var_length):
    """
    [description]
    : resample a CSI sequence along time to var_length (antialiased polyphase resampling). Shorter
      sequences are left zero-padded, the same convention load_data_x uses.
    """
    var_num = var_x.shape[0]
    if var_num == var_length:
        return var_x.astype(np.float32)
    if var_num < var_length:
        var_out = np.zeros((var_length, var_x.shape[1]), dtype=np.float32)
        var_out[var_length - var_num:] = var_x
        return var_out
    return resample_poly(var_x, var_length, var_num, axis=0).astype(np.float32)


def load_wimans_room(var_room, var_max_samples=None, var_length=None, var_band=None, var_users=None):
    """
    [description]
    : load one WiMANS room as multi-user (CSI, position-set) samples. A sample's label set holds the
      occupied discrete locations, one per user, in the room's [0, 1]^2 frame; an empty-room sample
      (count 0) is kept with an all-zero mask, so "no person" is a valid target.
    : var_length: window length in timesteps (defaults to preset["hwild"]["window"])
    : var_max_samples: int or None, cap the number of samples by a fixed-seed random subset
    : return: (X (num_samples, length, 90) float32, POS (num_samples, MAX_USERS, 2) normalized,
      MASK (num_samples, MAX_USERS) 0/1)
    """
    #
    var_length = var_length if var_length is not None else preset["hwild"]["window"]
    var_band = var_band if var_band is not None else preset["data"]["wifi_band"]
    var_users = var_users if var_users is not None else preset["data"]["num_users"]
    var_pd = load_data_y(preset["path"]["data_y"], [var_room], var_band, var_users)
    var_layout = preset["layouts"][var_room]
    var_cols = [f"user_{var_idx}_location" for var_idx in range(1, 7)]
    #
    var_labels, var_sets = [], []
    for _, var_row in var_pd.iterrows():
        var_pts = [var_layout[var_row[var_col]] for var_col in var_cols if var_row[var_col] in var_layout]
        var_labels.append(var_row["label"])
        var_sets.append(var_pts)  # may be empty (empty room, count 0)
    if var_max_samples is not None and len(var_labels) > var_max_samples:
        ## WiMANS is ordered single-user first, so a "first N" cut would drop every multi-user
        ## sample; draw a fixed-seed random subset instead.
        var_keep = np.sort(np.random.RandomState(39).choice(len(var_labels), var_max_samples, replace=False))
        var_labels = [var_labels[var_idx] for var_idx in var_keep]
        var_sets = [var_sets[var_idx] for var_idx in var_keep]
    if not var_labels:
        raise ValueError(f"no WiMANS samples for room {var_room!r}")
    #
    var_num = len(var_labels)
    var_pos = np.zeros((var_num, MAX_USERS, 2), dtype=np.float32)
    var_mask = np.zeros((var_num, MAX_USERS), dtype=np.float32)
    for var_idx, var_pts in enumerate(var_sets):
        var_pts = np.asarray(var_pts, dtype=np.float32).reshape(-1, 2)[:MAX_USERS]
        if var_pts.size == 0:
            continue  # empty room: all-zero mask, no position
        var_pos[var_idx, :len(var_pts)] = var_pts
        var_mask[var_idx, :len(var_pts)] = 1.0
    var_pos = normalize_xy_meters(var_pos * 10.0, WIMANS_BOUNDS)  # layout units -> meters -> frame
    #
    var_x = np.empty((var_num, var_length, WIMANS_FEATURES), dtype=np.float32)
    for var_idx, var_label in enumerate(var_labels):
        var_amp = np.load(os.path.join(preset["path"]["data_x"], var_label + ".npy")).astype(np.float32)
        ## (time, 3, 3, 30): average the 3 TX antennas -> (time, 3, 30) -> (time, 90)
        var_amp = var_amp.mean(axis=2).reshape(var_amp.shape[0], -1)
        var_x[var_idx] = _resample_time(var_amp, var_length)
    #
    return var_x, var_pos, var_mask


def _hwild_room_sets(var_room, var_length=None, var_max_samples=None, **kwargs):
    """
    [description]
    : load one H-WILD room as one-element position sets (the person's UWB position), padded to
      MAX_USERS with the mask marking the single valid entry.
    """
    var_x, var_xy, _ = load_hwild_room(var_room, var_window=var_length,
                                       var_max_windows=var_max_samples, **kwargs)
    var_num = len(var_x)
    var_pos = np.zeros((var_num, MAX_USERS, 2), dtype=np.float32)
    var_mask = np.zeros((var_num, MAX_USERS), dtype=np.float32)
    var_pos[:, 0] = var_xy
    var_mask[:, 0] = 1.0
    return var_x, var_pos, var_mask


def load_unified_room(var_room, var_max_samples=None, var_length=None, **kwargs):
    """
    [description]
    : load any room of either dataset as (X (N, length, 90), POS (N, MAX_USERS, 2) normalized,
      MASK (N, MAX_USERS)).
    : var_max_samples: cap on samples/windows per room
    : kwargs: forwarded to the H-WILD loader (e.g. var_root, var_users, var_interference, var_max_files)
    """
    #
    if var_room in WIMANS_ROOMS:
        return load_wimans_room(var_room, var_max_samples=var_max_samples, var_length=var_length)
    return _hwild_room_sets(var_room, var_length=var_length, var_max_samples=var_max_samples, **kwargs)


#
##
if __name__ == "__main__":
    #
    ##
    for var_room in unified_room_names():
        var_x, var_pos, var_mask = load_unified_room(var_room, var_max_samples=50)
        var_counts = var_mask.sum(axis=1)
        print(f"{dataset_of(var_room):6s} {var_room:12s} X {var_x.shape} POS {var_pos.shape} "
              f"span {tuple(round(v, 2) for v in room_span(var_room))} "
              f"users/sample min {int(var_counts.min())} max {int(var_counts.max())} "
              f"mean {var_counts.mean():.2f}")
