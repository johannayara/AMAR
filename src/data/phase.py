"""
[file]          phase.py
[description]   The CSI representation switch (amplitude vs phase) shared by every WiMANS loader,
                plus the sanitisation raw phase needs before it is usable.

                preset["data"]["csi_type"] selects the representation:
                  - "amplitude" (default): magnitude tensors (dataset/wifi_csi/amp/*.npy), unchanged.
                  - "phase": the phase tensors (dataset/wifi_csi/phase/*.npy, extracted from the raw
                    complex CSI with `python src/data/preprocess.py --what phase`). Every source's
                    amplitude directory "<...>/wifi_csi/amp" is read as its sibling "<...>/phase", so
                    the real, simulated and anywhere sources resolve through the same code path.

                Raw CSI phase is not physically meaningful: each frame carries a constant offset and a
                linear ramp across subcarriers from the packet-detection timing and the sampling-time /
                carrier-frequency offset. preset["data"]["phase"]["sanitize"] chooses what to do with
                it ("linear", the default, removes the ramp; "none" keeps the raw wrapped phase). The
                sanitisation is applied on load, so one extracted phase file serves either setting.
"""
#
##

import os

import numpy as np

from configs.preset import preset

#
##
CSI_TYPES = ("amplitude", "phase")
PHASE_SANITIZE_MODES = ("linear", "none", "raw")
NO_SANITIZE_MODES = ("none", "raw")
AMP_DIRNAME = "amp"
PHASE_DIRNAME = "phase"


def resolve_csi_type():
    """
    [description]
    : preset["data"]["csi_type"], validated. "amplitude" keeps the historical behaviour.
    """
    #
    var_type = (preset.get("data", {}).get("csi_type") or "amplitude").lower()
    if var_type not in CSI_TYPES:
        raise ValueError(f"preset['data']['csi_type'] must be one of {CSI_TYPES}; got {var_type!r}")
    return var_type


def resolve_phase_sanitize():
    """
    [description]
    : preset["data"]["phase"]["sanitize"], validated. Missing key => "linear"; an explicit None (which
      is what `--hp data.phase.sanitize=none` parses to) => "none". Only consulted in phase mode.
    """
    #
    var_phase_cfg = preset.get("data", {}).get("phase", {}) or {}
    var_mode = var_phase_cfg["sanitize"] if "sanitize" in var_phase_cfg else "linear"
    if var_mode is None:
        var_mode = "none"
    var_mode = str(var_mode).lower()
    if var_mode not in PHASE_SANITIZE_MODES:
        raise ValueError(f"preset['data']['phase']['sanitize'] must be one of {PHASE_SANITIZE_MODES}; "
                         f"got {var_mode!r}")
    return var_mode


def is_phase():
    """
    [description]
    : True when the run reads phase rather than amplitude.
    """
    #
    return resolve_csi_type() == "phase"


def csi_dir(var_amp_dir):
    """
    [description]
    : the directory holding the selected representation for a source whose amplitude directory is
      var_amp_dir. In amplitude mode this is a no-op. In phase mode the trailing "amp" segment is
      swapped for "phase" (an already-swapped directory is returned unchanged, so the call is
      idempotent), which is what keeps a source's two representations side by side.
    : var_amp_dir: string, the source's amplitude directory, e.g. "dataset/wifi_csi/amp"
    : return: string, the directory the loaders should read
    """
    #
    if resolve_csi_type() != "phase":
        return var_amp_dir
    var_head, var_tail = os.path.split(os.path.normpath(var_amp_dir))
    if var_tail == PHASE_DIRNAME:
        return var_amp_dir
    if var_tail != AMP_DIRNAME:
        raise ValueError(
            f"csi_type='phase' expects an amplitude directory ending in '{AMP_DIRNAME}'; got "
            f"{var_amp_dir!r}. Extract phase first (`python src/data/preprocess.py --what phase`).")
    var_phase_dir = os.path.join(var_head, PHASE_DIRNAME)
    if not os.path.isdir(var_phase_dir):
        raise FileNotFoundError(
            f"csi_type='phase' but the phase directory {var_phase_dir!r} does not exist; extract it "
            f"with `python src/data/preprocess.py --what phase`. Note only the real WiMANS source has "
            f"raw .mat captures -- the simulated and anywhere sources ship amplitude only, so disable "
            f"them (preset['data']['synth_wimans']) for a phase run.")
    return var_phase_dir


def sanitize_phase(var_phase, var_mode=None):
    """
    [description]
    : sanitise CSI phase along its subcarrier axis. "linear" unwraps the subcarrier axis then
      subtracts the per-frame linear trend (slope and offset)
    : var_phase: array (..., num_subcarriers), the subcarrier axis last (as (time, 3 TX, 3 RX, 30))
    : var_mode: "linear" (or "none"/"raw" to keep the input) | None 
    : return: float32 array, same shape
    """
    #
    if var_mode is None:
        var_mode = resolve_phase_sanitize()
    var_mode = str(var_mode).lower()
    if var_mode in NO_SANITIZE_MODES:
        return var_phase
    if var_mode != "linear":
        raise ValueError(f"unknown phase sanitize mode {var_mode!r}; choose from {PHASE_SANITIZE_MODES}")
    var_out = np.asarray(var_phase, dtype=np.float64)
    var_num_subcarriers = var_out.shape[-1]
    if var_num_subcarriers < 2:
        # a single subcarrier has no ramp to fit; unwrap/trend removal would divide by zero
        return var_out.astype(np.float32)
    var_out = np.unwrap(var_out, axis=-1)
    var_k = np.arange(var_num_subcarriers, dtype=np.float64)
    var_k_centered = var_k - var_k.mean()
    var_k_var = float((var_k_centered ** 2).sum())
    var_mean = var_out.mean(axis=-1, keepdims=True)
    var_slope = ((var_out - var_mean) * var_k_centered).sum(axis=-1, keepdims=True) / var_k_var
    return (var_out - (var_slope * var_k_centered + var_mean)).astype(np.float32)


def prepare_csi(var_x):
    """
    [description]
    : apply the selected representation's preprocessing to a freshly loaded array: phase gets the
      sanitisation (a no-op when it is off or when the representation is amplitude).
    : var_x: array as read from the resolved CSI directory
    : return: float32 array, same shape
    """
    #
    if resolve_csi_type() != "phase":
        return var_x
    return sanitize_phase(var_x)
