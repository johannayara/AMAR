#!/usr/bin/env python3
"""
Plot F1, Recall, Precision and PPS over epochs from AMAR training logs.

Automatically finds out_*.txt files in the output directory and overlays
all runs (repeats) on each subplot.

Usage:
    python plot_metrics.py                     # scans ./output by default
    python plot_metrics.py --dir results       # use a different folder
    python plot_metrics.py --out metrics.png   # save to file instead of showing
"""

import re
import os
import glob
import argparse
import matplotlib.pyplot as plt

# One regex per line we want to capture. The epoch comes from the header
# line; metrics are on their own labeled lines, in order.
EPOCH_RE = re.compile(
    r"--- Layer (layer_\d+) - Epoch (\d+)/\d+ ---"
)
TOTALS_RE = re.compile(
    r"Total Loss: (?P<total>[\d.]+) \| Class Loss: (?P<class_loss>[\d.]+) "
    r"\| RVQ Loss: (?P<rvq>[\d.]+) \| valid Loss: (?P<valid_loss>[\d.]+)"
)
ERROR_RE = re.compile(
    r"Total Error Train: (?P<err_train>[\d.]+) \| Total Error valid: (?P<err_valid>[\d.]+)"
)
PPS_RE = re.compile(
    r"Perfect Prediction % Train: (?P<pps_train>[\d.]+) \| "
    r"Perfect Prediction % valid: (?P<pps_valid>[\d.]+)"
)
ACC_RE = re.compile(
    r"Accuracy Train: (?P<acc_train>[\d.]+) \| Accuracy valid: (?P<acc_valid>[\d.]+)"
)
METRICS_RE = re.compile(
    r"Precision: (?P<precision>[\d.]+) \| Recall: (?P<recall>[\d.]+) "
    r"\| F1 Score: (?P<f1>[\d.]+)"
)


def natural_key(path):
    """Sort out_0.txt < out_1.txt < ... < out_10.txt."""
    m = re.search(r"out_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_log_files(directory):
    """Return all out_*.txt files in directory, naturally sorted."""
    files = glob.glob(os.path.join(directory, "out_*.txt"))
    return sorted(files, key=natural_key)


def parse_log(path):
    """Parse one AMAR output file into a list of per-epoch dicts."""
    epochs = []
    current = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if m:
                # flush previous epoch block
                if current is not None:
                    epochs.append(current)
                current = {
                    "layer": m.group(1),
                    "epoch": int(m.group(2)),
                }
                continue

            if current is None:
                continue

            for regex in (TOTALS_RE, ERROR_RE, PPS_RE, ACC_RE, METRICS_RE):
                m = regex.match(line.strip())
                if m:
                    current.update(
                        {k: float(v) for k, v in m.groupdict().items()}
                    )
                    break

    if current is not None:
        epochs.append(current)

    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="../output",
                        help="Directory containing out_*.txt logs (default: ./output)")
    parser.add_argument("--out", default="../figures", help="Save figure to this path")
    parser.add_argument("--pps-scale", action="store_true",
                        help="Show PPS as 0-100 %% (default: 0-1 to match F1)")
    args = parser.parse_args()

    files = find_log_files(args.dir)
    if not files:
        raise SystemExit(
            f"No out_*.txt files found in '{args.dir}'. "
            f"Check the path or pass --dir <folder>."
        )
    print(f"Found {len(files)} log file(s) in '{args.dir}':")
    for f in files:
        print(f"  {f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    (ax_f1, ax_prec), (ax_rec, ax_pps) = axes
    fig.suptitle('AMAR metrics over epochs', fontsize=16)
    pps_scale = 100.0 if args.pps_scale else 1.0

    metrics = [
        (ax_f1,    "f1",        "F1 Score"),
        (ax_prec,  "precision", "Precision"),
        (ax_rec,   "recall",    "Recall"),
        (ax_pps,   "pps_valid", "PPS (valid)"),
    ]

    colors = [plt.cm.cool(i / max(len(files) - 1, 1)) for i in range(len(files))]

    for i, path in enumerate(files):
        epochs = parse_log(path)
        complete = [e for e in epochs if all(
            k in e for k in ("f1", "precision", "recall", "pps_valid"))]

        if not complete:
            print(f"WARNING: no complete epoch blocks in {path}, skipping")
            continue

        ep = [e["epoch"] for e in complete]
        label = f"Run {natural_key(path) + 1}"  # Run 1, Run 2, ...
        color = colors[i]

        for ax, key, title in metrics:
            vals = [e[key] for e in complete]
            if key == "pps_valid":
                vals = [v / 100.0 * pps_scale for v in vals]
            ax.plot(ep, vals, marker="o", markersize=3,
                    linewidth=1.5, color=color, label=label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        best = max(complete, key=lambda e: e["f1"])
        print(f"{label} ({os.path.basename(path)}): "
              f"{len(complete)} epochs parsed, "
              f"best F1 = {best['f1']:.4f} @ epoch {best['epoch']}")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.out:
        plt.savefig(args.out)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()