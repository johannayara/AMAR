"""
[file]          ntu_har.py
[description]   NTU-Fi HAR amplitude loader, with the layout of the SenseFi benchmark
                (https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark, dataset.py:CSI_Dataset):

                    <root>/train_amp/<class>/<class><i>.mat     936 captures
                    <root>/test_amp/<class>/<class><i>.mat      264 captures

                <class> is one of box, circle, clean, fall, run, walk, and every .mat holds the CSI
                amplitude under the key "CSIamp" as (3 antennas, 114 subcarriers, 2000 packets).

                The benchmark's preprocessing is reproduced exactly: standardise with the dataset's
                global amplitude mean/std, then decimate the packet axis 2000 -> 500 with x[:, ::4].

                The (antenna, subcarrier) plane is then flattened into the feature axis the
                AMAR_WO_RVQ backbone consumes. NTU-Fi HAR has 114 subcarriers while WiMANS has 30, so
                the subcarrier axis is resampled to features // 3; that is what lets one model consume
                both datasets. The joint task itself lives in src/data/activity_transfer.py.
"""
#
##

import glob
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from configs.preset import preset

#
##
NTU_HAR_ANTENNAS = 3
NTU_HAR_SUBCARRIERS = 114
NTU_HAR_PACKETS = 2000
NTU_HAR_DECIMATION = 4
NTU_HAR_LENGTH = NTU_HAR_PACKETS // NTU_HAR_DECIMATION  # 500 packets after decimation
NTU_HAR_FEATURES = 270  # 3 antennas x 90 subcarriers: the AMAR_WO_RVQ WiMANS feature count
NTU_HAR_MEAN = 42.3199  # global amplitude statistics of the benchmark's CSI_Dataset
NTU_HAR_STD = 4.9802
NTU_HAR_SPLITS = ["train_amp", "test_amp"]


def ntu_har_root():
    """
    [description]
    : dataset root holding the train_amp/ and test_amp/ folders (preset["ntu_har"]["path"]).
    """
    return preset.get("ntu_har", {}).get("path", "dataset/ntu_har")


def ntu_har_splits():
    """
    [description]
    : the splits a run draws from (preset["ntu_har"]["splits"]).
    """
    return list(preset.get("ntu_har", {}).get("splits", NTU_HAR_SPLITS))


def ntu_har_class_names(var_root=None, var_splits=None):
    """
    [description]
    : the class-folder names present in the requested splits. The benchmark's own loader indexes
      classes by glob order, which is filesystem-dependent; sorting keeps the class index
      reproducible across machines.
    : return: sorted list of class names
    """
    #
    ##
    var_root = var_root or ntu_har_root()
    var_splits = var_splits or ntu_har_splits()
    var_names = set()
    for var_split in var_splits:
        for var_dir in glob.glob(os.path.join(var_root, var_split, "*")):
            if os.path.isdir(var_dir):
                var_names.add(os.path.basename(var_dir))
    if not var_names:
        raise FileNotFoundError(
            f"no class folders under {os.path.join(var_root, var_splits[0])!r}. Expected the SenseFi "
            f"layout <root>/train_amp/<class>/<class><i>.mat and <root>/test_amp/<class>/<class><i>.mat")
    return sorted(var_names)


def read_ntu_har_sample(var_path):
    """
    [description]
    : one capture, standardised and decimated exactly as the benchmark's CSI_Dataset does.
    : var_path: path of a .mat file
    : return: (3, 114, 500) float32 amplitude
    """
    #
    ##
    try:
        var_mat = loadmat(var_path)
    except NotImplementedError:
        ## MATLAB v7.3 is HDF5, and the array comes back transposed (MATLAB is column-major).
        import h5py
        with h5py.File(var_path, "r") as var_file:
            if "CSIamp" not in var_file:
                raise KeyError(f"{var_path} has no 'CSIamp' dataset (keys: {list(var_file)})")
            var_x = np.asarray(var_file["CSIamp"]).T
    else:
        if "CSIamp" not in var_mat:
            raise KeyError(f"{var_path} has no 'CSIamp' variable "
                           f"(keys: {sorted(k for k in var_mat if not k.startswith('__'))})")
        var_x = np.asarray(var_mat["CSIamp"])
    #
    var_x = (var_x.astype(np.float32) - NTU_HAR_MEAN) / NTU_HAR_STD
    return np.ascontiguousarray(var_x[:, :, ::NTU_HAR_DECIMATION])


def harmonize_ntu_features(var_x, var_features):
    """
    [description]
    : (antennas, subcarriers, time) -> (time, features), resampling the subcarrier axis to
      features // antennas. A no-op when the subcarrier count already matches, so the default
      (3 x 114 = 342) is untouched and only a run that has to match another dataset's feature count
      resamples.
    : return: (time, features) float32
    """
    #
    ##
    var_out_subcarriers = var_features // var_x.shape[0]
    if var_features % var_x.shape[0]:
        raise ValueError(f"var_features={var_features} is not a multiple of "
                         f"{var_x.shape[0]} antennas")
    if var_x.shape[1] != var_out_subcarriers:
        var_x = resample_poly(var_x, var_out_subcarriers, var_x.shape[1], axis=1)
    var_x = np.ascontiguousarray(np.transpose(var_x, (2, 0, 1)))
    return var_x.reshape(var_x.shape[0], var_features).astype(np.float32)


def load_ntu_har(var_root=None, var_splits=None, var_classes=None, var_features=None,
                 var_max_files=None):
    """
    [description]
    : load NTU-Fi HAR captures and their class indices.
    : var_classes: class names to keep (default: every class folder found)
    : var_features: feature count of the returned windows (default preset["ntu_har"]["features"])
    : var_max_files: cap the .mat files read per class (quick runs / smoke tests)
    : return: (X (N, 500, features) float32, y (N,) int64 indices into the returned class list,
      var_class_names (list))
    """
    #
    ##
    var_root = var_root or ntu_har_root()
    var_splits = var_splits or ntu_har_splits()
    var_features = var_features or preset.get("ntu_har", {}).get("features", NTU_HAR_FEATURES)
    var_class_names = ntu_har_class_names(var_root, var_splits)
    if var_classes is not None:
        var_missing = sorted(set(var_classes) - set(var_class_names))
        if var_missing:
            raise ValueError(f"NTU-Fi HAR has no class(es) {var_missing}; available {var_class_names}")
        var_class_names = [var_name for var_name in var_class_names if var_name in set(var_classes)]
    #
    var_paths, var_labels = [], []
    for var_idx, var_class in enumerate(var_class_names):
        var_files = []
        for var_split in var_splits:
            var_files.extend(sorted(glob.glob(os.path.join(var_root, var_split, var_class, "*.mat"))))
        if var_max_files is not None:
            var_files = var_files[:var_max_files]
        if not var_files:
            var_split_list = ",".join(var_splits)
            raise FileNotFoundError(f"no .mat files for NTU-Fi HAR class {var_class!r} under "
                                    f"{var_root} (splits: {var_split_list})")
        var_paths.extend(var_files)
        var_labels.extend([var_idx] * len(var_files))
    #
    var_x = np.empty((len(var_paths), NTU_HAR_LENGTH, var_features), dtype=np.float32)
    for var_i, var_path in enumerate(var_paths):
        var_x[var_i] = harmonize_ntu_features(read_ntu_har_sample(var_path), var_features)
    #
    return var_x, np.asarray(var_labels, dtype=np.int64), var_class_names
