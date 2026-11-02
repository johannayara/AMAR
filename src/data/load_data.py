"""
[file]          load_data.py
[description]   load annotation file and CSI amplitude, and encode labels
"""
#
##
import os
import numpy as np
import pandas as pd
#
from configs.preset import preset
from src.data.csi_preprocess import preprocess_csi

#
##
def load_data_y(var_path_data_y,
                var_environment = None, 
                var_wifi_band = None, 
                var_num_users = None):
    """
    [description]
    : load annotation file (*.csv) as a pandas dataframe
    : according to selected environment(s), WiFi band(s), and number(s) of users
    [parameter]
    : var_path_data_y: string, path of annotation file
    : var_environment: list, selected environment(s), e.g., ["classroom"]
    : var_wifi_band: list, selected WiFi band(s), e.g., ["2.4"]
    : var_num_users: list, selected number(s) of users, e.g., ["0", "1", "2"]
    [return]
    : data_pd_y: pandas dataframe, labels of selected data
    """
    #
    ##
    data_pd_y = pd.read_csv(var_path_data_y, dtype = str)
    #
    if var_environment is not None:
        data_pd_y = data_pd_y[data_pd_y["environment"].isin(var_environment)]
    #
    if var_wifi_band is not None:
        data_pd_y = data_pd_y[data_pd_y["wifi_band"].isin(var_wifi_band)]
    #
    if var_num_users is not None:
        data_pd_y = data_pd_y[data_pd_y["number_of_users"].isin(var_num_users)]
    #
    return data_pd_y

#
##
def load_data_x(var_path_data_x, 
                var_label_list,
                var_preprocess = None):
    """
    [description]
    : load CSI amplitude (*.npy)
    according to a label list of selected data
    [parameter]
    : var_path_data_x: string, directory of CSI amplitude files
    : var_label_list: list, selected labels
    : var_preprocess: optional dict overriding preset["data"]["preprocess"] (see csi_preprocess); the
      AMAR_CSI_PREPROCESS env var also overrides. When any op is enabled the returned array is
      (N, length, feature) with the trailing axes flattened; otherwise it keeps the raw trailing shape.
    [return]
    : data_x: numpy array, CSI amplitude
    """
    #
    ##
    var_path_list = [os.path.join(var_path_data_x, var_label + ".npy") for var_label in var_label_list]
    #
    # Preallocate the output and write each sample in place. Building a Python list of padded
    # samples first would transiently hold a second full copy of the whole dataset in memory.
    var_length = preset["data"]["length"]
    data_x = None
    ## the true frame count of every sample, needed to mask the front zero-padding in preprocessing
    var_lengths = np.zeros(len(var_path_list), dtype = np.int64)
    #
    for var_idx, var_path in enumerate(var_path_list):
        #
        data_csi = np.load(var_path)
        #
        if data_x is None:
            data_x = np.zeros((len(var_path_list), var_length) + data_csi.shape[1:], dtype=np.float32)
        #
        var_lengths[var_idx] = data_csi.shape[0]
        var_pad_length = var_length - data_csi.shape[0]
        #
        if var_pad_length < 0:
            raise ValueError(
                f"{var_path} has {data_csi.shape[0]} timesteps, longer than "
                f"preset['data']['length']={var_length}"
            )
        #
        # Same leading zero-padding as np.pad(data_csi, ((var_pad_length, 0), ...))
        data_x[var_idx, var_pad_length:] = data_csi
    #
    if data_x is None:
        data_x = np.zeros((0, var_length), dtype=np.float32)
    #
    return preprocess_csi(data_x, var_lengths, var_preprocess)


def harmonize_wimans_antennas(var_amp, var_num_tx=3, var_num_rx=3):
    """
    [description]
    : put a WiMANS amplitude capture on the 3 x 3 antenna grid the repo's WiMANS paths assume.
      Real WiMANS is stored as (time, 3 TX, 3 RX, subcarriers) and is returned unchanged. A capture
      with a single antenna pair -- e.g. the simulated few-shot dataset
      (dataset/synth_wimans/fewshot-groups-wimans), which is (time, 1, 1, 30) -- is broadcast over
      the TX and RX axes, so a single-link source can be dropped into any WiMANS path with no change
      to the model geometry or the feature count. The copies carry no extra information: this makes
      the tensor the right *shape*, it does not add channel diversity the source does not have.
    : var_amp: (time, TX, RX, subcarriers) float32
    : return: (time, var_num_tx, var_num_rx, subcarriers) float32
    """
    #
    ##
    var_amp = np.asarray(var_amp, dtype=np.float32)
    if var_amp.ndim != 4:
        raise ValueError(f"WiMANS amplitude must be (time, TX, RX, subcarriers); got shape "
                         f"{var_amp.shape}")
    var_time, var_tx, var_rx = var_amp.shape[:3]
    if (var_tx, var_rx) == (var_num_tx, var_num_rx):
        return var_amp
    if var_tx == 1 and var_rx == 1:
        return np.ascontiguousarray(np.broadcast_to(
            var_amp, (var_time, var_num_tx, var_num_rx, var_amp.shape[3])))
    raise ValueError(
        f"WiMANS amplitude has {var_tx} TX x {var_rx} RX antennas; expected "
        f"{var_num_tx} x {var_num_rx} (real WiMANS) or 1 x 1 (a single-link capture such as the "
        f"simulated dataset). Resample the antenna axes upstream if the layout is something else.")


def wimans_sources():
    """
    [description]
    : every WiMANS-format source a run may load from, as (suffix, data_x, data_y, bands). The real
      WiMANS source is always first with an empty suffix and the preset's own paths/band; the
      simulated source (preset["data"]["synth_wimans"]) is appended when it is enabled, with its own
      paths and band. The two coexist, which is what lets a run be scored on the real rooms alone or
      on the real rooms plus the simulated ones.
    """
    #
    ##
    var_sources = [("", preset["path"]["data_x"], preset["path"]["data_y"],
                    list(preset["data"]["wifi_band"]))]
    var_synth = preset["data"].get("synth_wimans") or {}
    if var_synth.get("enabled"):
        var_sources.append((var_synth.get("suffix", "_synth"), var_synth["data_x"],
                            var_synth["data_y"],
                            list(var_synth.get("wifi_band") or preset["data"]["wifi_band"])))
    return var_sources


def wimans_room_names(var_environments=None):
    """
    [description]
    : every WiMANS room key a run may load: each environment name (real WiMANS), plus "<name><suffix>"
      for the simulated source when it is enabled. Requesting only the bare names scores on real
      WiMANS alone; requesting both scores on the two sources together, and each keeps its own key so
      the metrics can be read per source.
    """
    #
    ##
    var_environments = (list(var_environments) if var_environments is not None
                        else list(preset["data"]["environment"]))
    var_rooms = list(var_environments)
    for var_suffix, _, _, _ in wimans_sources():
        if var_suffix:
            var_rooms.extend(var_name + var_suffix for var_name in var_environments)
    return var_rooms


def wimans_room_source(var_room, var_environments=None):
    """
    [description]
    : resolve a WiMANS room key to (base_room, data_x, data_y, bands). "classroom" resolves to the
      real WiMANS source and base room "classroom"; "classroom_synth" to the simulated source and the
      same base room, which is what keeps the room's layout and its physical frame identical between
      the two sources.
    """
    #
    ##
    var_environments = (list(var_environments) if var_environments is not None
                        else list(preset["data"]["environment"]))
    var_sources = wimans_sources()
    for var_suffix, var_data_x, var_data_y, var_bands in var_sources:
        if not var_suffix and var_room in var_environments:
            return var_room, var_data_x, var_data_y, var_bands
        if var_suffix and var_room.endswith(var_suffix):
            var_base = var_room[:-len(var_suffix)]
            if var_base in var_environments:
                return var_base, var_data_x, var_data_y, var_bands
    raise ValueError(f"unknown WiMANS room {var_room!r}; expected one of "
                     f"{wimans_room_names(var_environments)}")

#
##
def encode_data_y(data_pd_y, 
                  var_task):
    """
    [description]
    : encode labels according to specific task
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_task: string, indicate task
    [return]
    : data_y: numpy array, label encoding of task
    """
    #
    ##
    if var_task == "identity":
        #
        data_y = encode_identity(data_pd_y)
    #
    elif var_task == "activity":
        #
        data_y = encode_activity(data_pd_y, preset["encoding"]["activity"])
    #
    elif var_task == "location":
        #
        data_y = encode_location(data_pd_y, preset["encoding"]["location"])
    #
    elif var_task == "count":
        #
        data_y = encode_count(data_pd_y)
    #
    return data_y

#
##
def encode_identity(data_pd_y):
    """
    [description]
    : encode identity labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    [return]
    : data_identity_onehot_y: numpy array, onehot encoding for identity labels
    """
    #
    ##
    data_location_pd_y = data_pd_y[["user_1_location", "user_2_location", 
                                    "user_3_location", "user_4_location", 
                                    "user_5_location", "user_6_location"]]
    # 
    data_identity_y = data_location_pd_y.to_numpy(copy = True).astype(str)
    #
    data_identity_y[data_identity_y != "nan"] = 1
    data_identity_y[data_identity_y == "nan"] = 0
    #
    data_identity_onehot_y = data_identity_y.astype("int8")
    #
    return data_identity_onehot_y

#
##
def encode_activity(data_pd_y, 
                    var_encoding):
    """
    [description]
    : encode activity labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_encoding: dict, encoding of different activities
    [return]
    : data_activity_onehot_y: numpy array, onehot encoding for activity labels
    """
    #
    ##
    data_activity_pd_y = data_pd_y[["user_1_activity", "user_2_activity", 
                                    "user_3_activity", "user_4_activity", 
                                    "user_5_activity", "user_6_activity"]]
    #
    data_activity_y = data_activity_pd_y.to_numpy(copy = True).astype(str)
    #
    data_activity_onehot_y = np.array([[var_encoding[var_y] for var_y in var_sample] for var_sample in data_activity_y])
    #
    return data_activity_onehot_y

#
##
def encode_location(data_pd_y, 
                    var_encoding):
    """
    [description]
    : encode location labels in a pandas dataframe
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_encoding: dict, encoding of different locations
    [return]
    : data_location_onehot_y: numpy array, onehot encoding for location labels
    """
    #
    ##
    data_location_pd_y = data_pd_y[["user_1_location", "user_2_location", 
                                    "user_3_location", "user_4_location", 
                                    "user_5_location", "user_6_location"]]
    #
    data_location_y = data_location_pd_y.to_numpy(copy = True).astype(str)
    #
    data_location_onehot_y = np.array([[var_encoding[var_y] for var_y in var_sample] for var_sample in data_location_y])
    #
    return data_location_onehot_y

#
##
def encode_count(data_pd_y):
    """
    [description]
    : encode the group-count label (number of people occupying the space) in a pandas dataframe.
      Unlike activity/location, this label is room-agnostic: it does not depend on where in the
      room each person is, only on how many people are present.
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    [return]
    : data_count_y: numpy array of shape (num_samples,), integer class index in [0, 5]
    """
    #
    ##
    data_count_y = data_pd_y["number_of_users"].to_numpy(copy = True).astype("int64")
    #
    return data_count_y

#
##
def encode_occupancy_y(data_pd_y,
                       var_environment):
    """
    [description]
    : encode the per-location occupancy used by the density-map model. Each person in WiMANS stands
      at one of the room's locations, so a sample is described by which of the locations are
      occupied. The occupancy is expressed in the shared, room-agnostic frame defined by
      preset["layouts"], which is what the density-map model predicts and renders.
    [parameter]
    : data_pd_y: pandas dataframe, labels of different tasks
    : var_environment: string, room name used to look up the location coordinates
    [return]
    : data_occupancy_y: numpy array of shape (num_samples, num_locations) with 0/1 entries,
      columns in the sorted order of the room's location keys
    """
    #
    ##
    var_layout = preset["layouts"][var_environment]
    var_names = sorted(var_layout)
    #
    ## A coordinate outside [0,1] would place the kernel off the grid and silently move the peak, so
    ## fail loudly instead. This catches e.g. coordinates left in cm after a layout redefinition.
    for var_letter, (var_cx, var_cy) in var_layout.items():
        if not (0.0 <= var_cx <= 1.0 and 0.0 <= var_cy <= 1.0):
            raise ValueError(
                f"layout coordinate for '{var_letter}' in '{var_environment}' is "
                f"({var_cx}, {var_cy}), outside the normalized [0,1] density grid")
    #
    var_locations = data_pd_y[["user_1_location", "user_2_location",
                               "user_3_location", "user_4_location",
                               "user_5_location", "user_6_location"]].to_numpy(copy = True).astype(str)
    #
    data_occupancy_y = np.zeros((len(data_pd_y), len(var_names)), dtype = np.float32)
    #
    for var_idx, var_sample in enumerate(var_locations):
        for var_letter in var_sample:
            if var_letter in var_layout:
                data_occupancy_y[var_idx, var_names.index(var_letter)] = 1.0
    #
    return data_occupancy_y

#
##
def test_load_data_y():
    """
    [description]
    : test load_data_y() function
    """
    #
    ##
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["classroom"]).describe())
    #
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["meeting_room"], 
                      var_wifi_band = ["2.4"]).describe())
    #
    print(load_data_y(preset["path"]["data_y"], 
                      var_environment = ["meeting_room"], 
                      var_wifi_band = ["2.4"], 
                      var_num_users = ["1", "2", "3"]).describe())

#
##
def test_load_data_x():
    """
    [description]
    : test load_data_x() function
    """
    #
    ##
    data_pd_y = load_data_y(preset["path"]["data_y"],
                            var_environment = ["meeting_room"], 
                            var_wifi_band = ["2.4"], 
                            var_num_users = None)
    #
    var_label_list = data_pd_y["label"].to_list()
    #
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    #
    print(data_x.shape)

#
##
def test_encode_identity():
    """
    [description]
    : test encode_identity() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_identity_onehot_y = encode_identity(data_pd_y)
    #
    print(data_identity_onehot_y.shape)
    #
    print(data_identity_onehot_y[2000])

#
##
def test_encode_activity():
    """
    [description]
    : test encode_activity() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_activity_onehot_y = encode_activity(data_pd_y, preset["encoding"]["activity"])
    #
    print(data_activity_onehot_y.shape)
    #
    print(data_activity_onehot_y[1560])

#
##
def test_encode_location():
    """
    [description]
    : test encode_location() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_location_onehot_y = encode_location(data_pd_y, preset["encoding"]["location"])
    #
    print(data_location_onehot_y.shape)
    #
    print(data_location_onehot_y[1560])

#
##
def test_encode_count():
    """
    [description]
    : test encode_count() function
    """
    #
    ##
    data_pd_y = pd.read_csv(preset["path"]["data_y"], dtype = str)
    #
    data_count_y = encode_count(data_pd_y)
    #
    print(data_count_y.shape)
    #
    print(np.bincount(data_count_y))

#
##
if __name__ == "__main__":
    #
    ##
    test_load_data_y()
    #
    test_load_data_x()
    #
    test_encode_identity()
    #
    test_encode_activity()
    #
    test_encode_location()
    #
    test_encode_count()