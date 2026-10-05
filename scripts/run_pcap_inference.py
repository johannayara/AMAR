"""
[file]          run_pcap_inference.py
[description]   Estimate people positions from a nexmon CSI pcap capture with a trained density model.

                Reads a nexmon CSI capture, turns it into the (time, feature) tensor the trained
                density model expects, runs the model and extracts the peaks of the predicted density
                map as the estimated positions.

                IMPORTANT: this is a best-effort transfer. The density model was trained on WiMANS
                (a different NIC, bandwidth and subcarrier count), so the capture's subcarrier axis is
                interpolated onto the model's feature axis and the amplitude is fed as-is. The output
                is a position estimate in the model's shared, TX-anchored normalized frame, not a
                metrically calibrated localization.
"""
#
##

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from configs.preset import preset
from src.data.nexmon import read_nexmon_pcap, build_model_input
from src.models.density_map import DensityMapNet, _extract_peaks
from src.utils import select_device


def parse_args():
    """
    [description]
    : parse arguments from input
    """
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--pcap", required=True, type=str, help="path of the nexmon CSI pcap capture")
    var_args.add_argument("--checkpoint", required=True, type=str,
                          help="path of a density-model checkpoint saved by the density runners "
                               "(model.pth / student_model.pth)")
    var_args.add_argument("--out", default="output/pcap_inference", type=str, help="output directory")
    var_args.add_argument("--peak_threshold", default=preset["density"]["peak_threshold"], type=float,
                          help="a map local maximum counts as a person when it exceeds this fraction "
                               "of the map maximum")
    var_args.add_argument("--peak_abs_floor", default=preset["density"].get("peak_abs_floor", 0.0),
                          type=float,
                          help="absolute floor: a map whose maximum is below it has no peaks, and no "
                               "cell below it counts (guards against spurious peaks on a near-empty map)")
    var_args.add_argument("--scale", default=None, type=float,
                          help="optional multiplier applied to the captured amplitude")
    var_args.add_argument("--max_frames", default=None, type=int, help="cap on the number of CSI frames")
    var_args.add_argument("--extent_m", default=None, type=float,
                          help="optional side length in meters of the normalized [0,1]^2 frame, used to "
                               "also report positions in meters")
    return var_args.parse_args()


def load_density_model(var_checkpoint_path, var_device):
    """
    [description]
    : rebuild a DensityMapNet from a checkpoint saved by save_density_checkpoint() and load its weights.
    """
    var_checkpoint = torch.load(var_checkpoint_path, map_location=var_device)
    var_layout_name = var_checkpoint["layout_name"]
    var_layout = preset["layouts"].get(var_layout_name)
    if var_layout is None:
        var_layout = preset["layouts"][preset["data"]["environment"][0]]
        print(f"WARNING: layout '{var_layout_name}' is not in preset['layouts']; using "
              f"'{preset['data']['environment'][0]}' for the (unused at inference) kernels")
    var_model = DensityMapNet(tuple(var_checkpoint["x_shape"]), var_layout,
                              grid_size=var_checkpoint["grid_size"],
                              sigma=var_checkpoint["sigma"],
                              hidden_dim=var_checkpoint.get("decoder_hidden"),
                              dropout=var_checkpoint.get("decoder_dropout"),
                              var_output_mode=var_checkpoint.get("output_mode", "probability")).to(var_device)
    var_model.load_state_dict(var_checkpoint["model_state_dict"])
    var_model.eval()
    return var_model, var_checkpoint


def visualize(var_density, var_peaks, var_save_path, var_tag, var_meta):
    """
    [description]
    : save the predicted density map with the detected positions circled.
    """
    import matplotlib.pyplot as plt
    #
    var_fig, var_ax = plt.subplots(figsize=(6, 6))
    var_ax.imshow(var_density, origin="upper", cmap="viridis")
    if len(var_peaks):
        var_ax.scatter(var_peaks[:, 0] * var_density.shape[1] - 0.5,
                       var_peaks[:, 1] * var_density.shape[0] - 0.5,
                       s=90, facecolors="none", edgecolors="red", linewidths=1.5)
    var_ax.set_xticks([])
    var_ax.set_yticks([])
    var_ax.set_title(f"{var_tag} - {len(var_peaks)} peaks "
                     f"({var_meta['num_frames']} frames x {var_meta['num_subcarriers']} subcarriers)")
    var_fig.tight_layout()
    var_out_path = os.path.join(var_save_path, f"pcap_inference_{var_tag}.png")
    var_fig.savefig(var_out_path, dpi=120)
    plt.close(var_fig)
    return var_out_path


#
##
def run():
    """
    [description]
    : read a nexmon CSI capture, run the trained density model and report the estimated positions.
    """
    #
    var_args = parse_args()
    var_device = select_device()
    print(f"Using device: {var_device}")
    #
    var_model, var_checkpoint = load_density_model(var_args.checkpoint, var_device)
    var_x_shape = tuple(var_checkpoint["x_shape"])
    print(f"Model input shape {var_x_shape} - layout '{var_checkpoint['layout_name']}'")
    #
    ## ============================================ Read capture ============================================
    #
    var_amplitude, var_meta = read_nexmon_pcap(var_args.pcap, var_max_frames=var_args.max_frames)
    print(f"Capture: {var_meta['num_frames']} frames x {var_meta['num_subcarriers']} subcarriers, "
          f"chip 0x{var_meta['chip']:04x}, rssi {var_meta.get('rssi')}, "
          f"core/spatial-stream {sorted(var_meta['core_spatial_stream_counts'])}")
    var_input = build_model_input(var_amplitude, var_x_shape, var_scale=var_args.scale)
    #
    ## ============================================ Predict ============================================
    #
    with torch.no_grad():
        var_density, var_count, _ = var_model(torch.from_numpy(var_input).unsqueeze(0).to(var_device))
    var_density = var_density[0].cpu().numpy()
    #
    ## positions are the peaks of the map, in the shared TX-anchored normalized frame
    var_peaks = _extract_peaks(var_density, var_args.peak_threshold, var_args.peak_abs_floor)
    #
    var_positions = []
    for var_peak in var_peaks:
        var_entry = {"x_norm": float(var_peak[0]), "y_norm": float(var_peak[1])}
        if var_args.extent_m is not None:
            var_entry["x_m"] = float(var_peak[0] * var_args.extent_m)
            var_entry["y_m"] = float(var_peak[1] * var_args.extent_m)
        var_positions.append(var_entry)
    #
    ## ============================================ Report ============================================
    #
    os.makedirs(var_args.out, exist_ok=True)
    var_tag = os.path.splitext(os.path.basename(var_args.pcap))[0]
    var_fig_path = visualize(var_density, var_peaks, var_args.out, var_tag, var_meta)
    #
    var_result = {
        "pcap": var_args.pcap,
        "checkpoint": var_args.checkpoint,
        "layout": var_checkpoint["layout_name"],
        "num_people_estimate": len(var_peaks),
        "soft_map_mass": float(var_count.item()),
        "peak_threshold": var_args.peak_threshold,
        "peak_abs_floor": var_args.peak_abs_floor,
        "extent_m": var_args.extent_m,
        "positions": var_positions,
        "capture": {k: var_meta[k] for k in
                    ("num_frames", "num_subcarriers", "chip", "rssi", "core_spatial_stream_counts")},
    }
    var_json_path = os.path.join(var_args.out, f"pcap_inference_{var_tag}.json")
    with open(var_json_path, "w") as var_file:
        json.dump(var_result, var_file, indent=4, default=str)
    #
    print(f"\nEstimated {len(var_peaks)} people:")
    for var_entry in var_positions:
        var_line = f"  x={var_entry['x_norm']:.3f}, y={var_entry['y_norm']:.3f} (normalized)"
        if "x_m" in var_entry:
            var_line += f"  |  x={var_entry['x_m']:.2f} m, y={var_entry['y_m']:.2f} m"
        print(var_line)
    print(f"\nFigure: {var_fig_path}\nJSON:   {var_json_path}")


if __name__ == "__main__":
    run()
