"""
[file]          preprocess.py
[description]   preprocess WiFi CSI data
"""
#
##
import os
import argparse
import numpy as np
import scipy.io as scio

#
##
def mat_to_csi(data_mat):
    """
    [description]
    : the raw complex CSI of one *.mat capture. Each element of the MATLAB "trace" struct carries the
      per-frame complex CSI in its last field ((time, 3 TX, 3 RX, 30) on the WiMANS captures).
    : data_mat: dict, raw WiFi CSI data from *.mat files
    : return: data_csi: complex array (time, TX, RX, subcarriers), the .mat's native precision
    """
    #
    ##
    var_length = data_mat["trace"].shape[0]
    #
    data_csi = [data_mat["trace"][var_t][0][0][0][-1] for var_t in range(var_length)]
    #
    return np.asarray(data_csi)

#
##
def mat_to_amp(data_mat):
    """
    [description]
    : calculate amplitude of raw WiFi CSI data
    [parameter]
    : data_mat: dict, raw WiFi CSI data from *.mat files
    [return]
    : data_csi_amp: numpy array, CSI amplitude
    """
    #
    ##
    return np.abs(mat_to_csi(data_mat)).astype(np.float32)

#
##
def mat_to_phase(data_mat):
    """
    [description]
    : calculate the (wrapped) phase of raw WiFi CSI data. Stored unwrapped only in the sense that the
      loader's sanitisation unwraps it; this is the raw angle in [-pi, pi], so one extracted file serves
      every preset["data"]["phase"]["sanitize"] setting.
    : data_mat: dict, raw WiFi CSI data from *.mat files
    : return: data_csi_phase: float32 array (time, TX, RX, subcarriers)
    """
    #
    ##
    return np.angle(mat_to_csi(data_mat)).astype(np.float32)

#
##
def extract_csi_amp(var_dir_mat, 
                    var_dir_amp):
    """
    [description]
    : read raw WiFi CSI files (*.mat), calcuate CSI amplitude, and save amplitude (*.npy)
    [parameter]
    : var_dir_mat: string, directory to read raw WiFi CSI files (*.mat)
    : var_dir_amp: string, directory to save WiFi CSI amplitude (*.npy)
    """
    #
    ##
    var_path_mat = os.listdir(var_dir_mat)
    #
    for var_c, var_path in enumerate(var_path_mat):
        #
        data_mat = scio.loadmat(os.path.join(var_dir_mat, var_path))
        #
        data_csi_amp = mat_to_amp(data_mat)
        #
        print(var_c, data_csi_amp.shape)
        #
        var_path_save = os.path.join(var_dir_amp, var_path.replace(".mat", ".npy"))
        #
        with open(var_path_save, "wb") as var_file:
            np.save(var_file, data_csi_amp)

#
##
def extract_csi_phase(var_dir_mat, 
                      var_dir_phase):
    """
    [description]
    : read raw WiFi CSI files (*.mat), calculate the (wrapped) CSI phase, and save phase (*.npy). The
      phase directory mirrors the amplitude one, so preset["data"]["csi_type"]="phase" resolves it by
      swapping the trailing "amp" segment (src/data/phase.py:csi_dir).
    : var_dir_mat: string, directory to read raw WiFi CSI files (*.mat)
    : var_dir_phase: string, directory to save WiFi CSI phase (*.npy)
    """
    #
    ##
    os.makedirs(var_dir_phase, exist_ok=True)
    var_path_mat = os.listdir(var_dir_mat)
    #
    for var_c, var_path in enumerate(var_path_mat):
        #
        data_mat = scio.loadmat(os.path.join(var_dir_mat, var_path))
        #
        data_csi_phase = mat_to_phase(data_mat)
        #
        print(var_c, data_csi_phase.shape)
        #
        var_path_save = os.path.join(var_dir_phase, var_path.replace(".mat", ".npy"))
        #
        with open(var_path_save, "wb") as var_file:
            np.save(var_file, data_csi_phase)

#
##
def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--dir_mat", default = "dataset/wifi_csi/mat", type = str)
    var_args.add_argument("--dir_amp", default = "dataset/wifi_csi/amp", type = str)
    var_args.add_argument("--dir_phase", default = "dataset/wifi_csi/phase", type = str)
    var_args.add_argument("--what", default = "amp", choices = ("amp", "phase", "both"),
                          type = str,
                          help="which representation to extract: 'amp' (default, unchanged), 'phase', "
                               "or 'both'")
    #
    return var_args.parse_args()

#
##
if __name__ == "__main__":
    #
    ##
    var_args = parse_args()
    #
    if var_args.what in ("amp", "both"):
        extract_csi_amp(var_dir_mat = var_args.dir_mat, 
                        var_dir_amp = var_args.dir_amp)
    if var_args.what in ("phase", "both"):
        extract_csi_phase(var_dir_mat = var_args.dir_mat, 
                          var_dir_phase = var_args.dir_phase)