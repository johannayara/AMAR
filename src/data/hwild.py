"""
[file]          hwild.py
[description]   Loader for the H-WILD human-held-device WiFi localization dataset
                (https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset).

                Each H-WILD *.mat capture holds one room / access point / volunteer / interference
                condition. It stores the CSI of every received packet as 3 antennas x 30 subcarriers
                (the Intel 5300 layout, same family as WiMANS) plus a UWB ground-truth position per
                packet. This module turns a capture into fixed-length CSI windows, each labelled with
                the mean UWB position over the window, and normalizes every position into a fixed
                per-room [0, 1]^2 frame.

                The frame is the coordinate grid of the dataset's own obtain_parameters.m (the grid
                the shipped triangulation demo searches over), transcribed into HWILD_ROOMS below.
                It is room geometry, not a function of the capture coordinates, so building the frame
                for the held-out room uses no labels of that room: the leave-one-room-out protocol
                stays clean. A normalized coordinate (u, v) maps back to meters with the room's
                (x_range, y_range), i.e. x = x_min + u * (x_max - x_min).

                The .mat files are MATLAB v7.3 (HDF5); scipy.io.loadmat cannot read them, so h5py is
                required. A v7 fallback is kept for captures converted by other tooling.
"""
#
##

import os
import glob

import numpy as np

from configs.preset import preset

#
##
## Per-room geometry, transcribed from the dataset's obtain_parameters.m. "dir" is the on-disk
## folder, "prefix" the room token inside the file names (e.g. Conference files start with "Con"),
## "x_range"/"y_range" the UWB coordinate grid of the room in meters, and "aps" the access points
## with captures (the lounge swaps sRE22 for sRE4). All values are raw dataset metadata.
HWILD_ROOMS = {
    "Conference": {
        "dir": "Conference",
        "prefix": "Con",
        "x_range": (-3.0, 7.0),
        "y_range": (-3.0, 7.0),
        "aps": ("sRE22", "sRE5", "sRE6", "sRE7"),
    },
    "Laboratory": {
        "dir": "Laboratory",
        "prefix": "Lab",
        "x_range": (-2.0, 10.0),
        "y_range": (-2.0, 10.0),
        "aps": ("sRE22", "sRE5", "sRE6", "sRE7"),
    },
    "Office": {
        "dir": "Office",
        "prefix": "Office",
        "x_range": (-3.0, 7.0),
        "y_range": (-4.0, 12.0),
        "aps": ("sRE22", "sRE5", "sRE6", "sRE7"),
    },
    "Lounge": {
        "dir": "Lounge",
        "prefix": "Lounge",
        "x_range": (-4.0, 12.0),
        "y_range": (-2.0, 14.0),
        "aps": ("sRE4", "sRE5", "sRE6", "sRE7"),
    },
}


def room_names():
    """
    [description]
    : names of the four H-WILD rooms, in a fixed order.
    """
    return list(HWILD_ROOMS)


def room_span(var_room):
    """
    [description]
    : the physical extent (x_span, y_span) of a room's frame in meters. Multiplying a normalized
      error component by it gives the error in meters on that axis.
    """
    var_x_range = HWILD_ROOMS[var_room]["x_range"]
    var_y_range = HWILD_ROOMS[var_room]["y_range"]
    return (var_x_range[1] - var_x_range[0], var_y_range[1] - var_y_range[0])


def normalize_xy(var_xy, var_room):
    """
    [description]
    : map raw UWB (x, y) in meters into the room's fixed [0, 1]^2 frame.
    : var_xy: numpy array (..., 2) of raw coordinates in meters
    : return: same shape, normalized
    """
    var_x_range = HWILD_ROOMS[var_room]["x_range"]
    var_y_range = HWILD_ROOMS[var_room]["y_range"]
    var_out = np.array(var_xy, dtype=np.float32, copy=True)
    var_out[..., 0] = (var_out[..., 0] - var_x_range[0]) / (var_x_range[1] - var_x_range[0])
    var_out[..., 1] = (var_out[..., 1] - var_y_range[0]) / (var_y_range[1] - var_y_range[0])
    return var_out


def denormalize_xy(var_xy, var_room):
    """
    [description]
    : inverse of normalize_xy: map a normalized (x, y) back to raw meters in the room's frame.
    """
    var_x_range = HWILD_ROOMS[var_room]["x_range"]
    var_y_range = HWILD_ROOMS[var_room]["y_range"]
    var_out = np.array(var_xy, dtype=np.float32, copy=True)
    var_out[..., 0] = var_out[..., 0] * (var_x_range[1] - var_x_range[0]) + var_x_range[0]
    var_out[..., 1] = var_out[..., 1] * (var_y_range[1] - var_y_range[0]) + var_y_range[0]
    return var_out


def parse_hwild_name(var_path):
    """
    [description]
    : parse an H-WILD file name "<prefix>_<ap>_user<id>_<w|wo>.mat" into its parts.
    : return: dict with keys ap, user, interference ("w" or "wo"), or None when it does not match
    """
    var_stem = os.path.splitext(os.path.basename(var_path))[0]
    var_parts = var_stem.split("_")
    if len(var_parts) != 4 or not var_parts[2].startswith("user"):
        return None
    return {"ap": var_parts[1], "user": var_parts[2][len("user"):], "interference": var_parts[3]}


def _read_hwild_mat(var_path):
    """
    [description]
    : read one H-WILD capture and return its CSI amplitude and raw UWB positions.
    : var_path: str, path of the *.mat file (MATLAB v7.3 / HDF5)
    : return: (amplitude (num_packets, 90) float32, xy (num_packets, 2) float32 in meters)
    """
    #
    ##
    try:
        import h5py
        with h5py.File(var_path, "r") as var_file:
            var_csi = var_file["features_csi"][:]
            var_x = np.asarray(var_file["uwb_coordinate_x"]).reshape(-1)
            var_y = np.asarray(var_file["uwb_coordinate_y"]).reshape(-1)
    except (OSError, ImportError):
        ## fallback for v7 captures converted by other tooling
        import scipy.io as scio
        var_mat = scio.loadmat(var_path)
        var_csi = np.asarray(var_mat["features_csi"])
        var_x = np.asarray(var_mat["uwb_coordinate_x"]).reshape(-1)
        var_y = np.asarray(var_mat["uwb_coordinate_y"]).reshape(-1)
    #
    ## features_csi is (3 antennas x 30 subcarriers, num_packets) with fields real/imag; the demo's
    ## reshape(specific_csi, 30, 3) confirms the antenna-major subcarrier layout. Amplitude is the
    ## complex modulus, transposed to the (time, feature) convention the WiMANS loader uses.
    var_real = np.asarray(var_csi["real"], dtype=np.float64)
    var_imag = np.asarray(var_csi["imag"], dtype=np.float64)
    var_amp = np.sqrt(var_real ** 2 + var_imag ** 2).T.astype(np.float32)
    #
    var_xy = np.stack([var_x, var_y], axis=1).astype(np.float32)
    if var_amp.shape[0] != var_xy.shape[0]:
        raise ValueError(f"{var_path}: {var_amp.shape[0]} CSI packets vs {var_xy.shape[0]} positions")
    #
    return var_amp, var_xy


def build_windows(var_amp, var_xy, var_window, var_stride):
    """
    [description]
    : slice one capture into fixed-length windows. The label of a window is the mean normalized
      position over its packets, i.e. where the person is across the window.
    : var_amp: (num_packets, num_features) float32
    : var_xy: (num_packets, 2) float32 normalized positions
    : return: (X (num_windows, window, features), XY (num_windows, 2)) or (None, None) when the
      capture is shorter than one window
    """
    #
    ##
    var_num_packets = var_amp.shape[0]
    if var_num_packets < var_window:
        return None, None
    var_num = 1 + (var_num_packets - var_window) // var_stride
    #
    var_x = np.empty((var_num, var_window, var_amp.shape[1]), dtype=np.float32)
    var_y = np.empty((var_num, 2), dtype=np.float32)
    for var_idx in range(var_num):
        var_start = var_idx * var_stride
        var_x[var_idx] = var_amp[var_start:var_start + var_window]
        var_y[var_idx] = var_xy[var_start:var_start + var_window].mean(axis=0)
    #
    return var_x, var_y


def list_hwild_files(var_room, var_root=None, var_aps=None, var_users=None, var_interference=None):
    """
    [description]
    : list the captures of a room matching the requested access points / volunteers / interference.
    : var_root: str or None, dataset root holding <dir>/*.mat (defaults to preset["hwild"]["path"])
    : return: list of paths, sorted
    """
    #
    ##
    var_root = var_root if var_root is not None else preset["hwild"]["path"]
    var_paths = sorted(glob.glob(os.path.join(var_root, HWILD_ROOMS[var_room]["dir"], "*.mat")))
    var_kept = []
    for var_path in var_paths:
        var_meta = parse_hwild_name(var_path)
        if var_meta is None:
            continue
        if var_aps is not None and var_meta["ap"] not in var_aps:
            continue
        if var_users is not None and var_meta["user"] not in var_users:
            continue
        if var_interference is not None and var_meta["interference"] not in var_interference:
            continue
        var_kept.append(var_path)
    return var_kept


def load_hwild_room(var_room, var_root=None, var_aps=None, var_users=None, var_interference=None,
                    var_window=None, var_stride=None, var_max_files=None, var_max_windows=None):
    """
    [description]
    : load one H-WILD room as a windowed (CSI, position) set.
    : var_window / var_stride: window length and hop in packets (defaults from preset["hwild"])
    : var_max_files: int or None, keep only the first N captures (quick runs / smoke tests)
    : var_max_windows: int or None, cap the number of windows per room
    : return: (X (num_windows, window, 90) float32, XY (num_windows, 2) float32 normalized, meta dict)
    """
    #
    ##
    var_window = var_window if var_window is not None else preset["hwild"]["window"]
    var_stride = var_stride if var_stride is not None else preset["hwild"]["stride"]
    var_paths = list_hwild_files(var_room, var_root, var_aps, var_users, var_interference)
    if var_max_files is not None:
        var_paths = var_paths[:var_max_files]
    if not var_paths:
        raise FileNotFoundError(
            f"no H-WILD captures for room {var_room!r} under "
            f"{(var_root if var_root is not None else preset['hwild']['path'])!r}")
    #
    ##
    var_x_list, var_y_list, var_names = [], [], []
    for var_path in var_paths:
        var_amp, var_xy = _read_hwild_mat(var_path)
        var_xy_norm = normalize_xy(var_xy, var_room)
        var_x, var_y = build_windows(var_amp, var_xy_norm, var_window, var_stride)
        if var_x is None:
            continue
        var_x_list.append(var_x)
        var_y_list.append(var_y)
        var_names.append(os.path.basename(var_path))
    #
    if not var_x_list:
        raise ValueError(
            f"no {var_room!r} capture is at least {var_window} packets long; lower "
            f"preset['hwild']['window']")
    var_x = np.concatenate(var_x_list)
    var_y = np.concatenate(var_y_list)
    #
    if var_max_windows is not None and len(var_x) > var_max_windows:
        ## deterministic thinning, so a capped run is reproducible
        var_keep = np.linspace(0, len(var_x) - 1, var_max_windows).astype(int)
        var_x, var_y = var_x[var_keep], var_y[var_keep]
    #
    var_meta = {"room": var_room, "num_files": len(var_names), "files": var_names,
                "window": var_window, "stride": var_stride,
                "span": room_span(var_room)}
    return var_x, var_y, var_meta


#
##
if __name__ == "__main__":
    #
    ##
    import sys
    var_root = sys.argv[1] if len(sys.argv) > 1 else preset["hwild"]["path"]
    for var_room in room_names():
        try:
            var_x, var_y, var_meta = load_hwild_room(var_room, var_root=var_root, var_max_files=2)
            print(f"{var_room}: X {var_x.shape} XY {var_y.shape} "
                  f"u [{var_y[:, 0].min():.3f}, {var_y[:, 0].max():.3f}] "
                  f"v [{var_y[:, 1].min():.3f}, {var_y[:, 1].max():.3f}] "
                  f"span {var_meta['span']} files {var_meta['num_files']}")
        except Exception as var_err:
            print(f"{var_room}: {var_err}")
