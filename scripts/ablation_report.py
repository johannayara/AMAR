#!/usr/bin/env python3
"""
[file]          ablation_report.py
[description]   Collect every experimental result artifact under output/ and answer the cross-domain
                ablation questions from what is actually on disk, then flag the artifacts that were
                produced by an older code version and must not be compared against current runs.

                Pure text reader: no torch, no dataset, no training. Every artifact is a `.txt` report
                (save_run_outputs) or a `.log` (a shell script's redirected stdout that ends with the
                same report block), so the whole analysis is regex parsing plus tables.

                Usage:
                    python scripts/ablation_report.py
                    python scripts/ablation_report.py --output-dir output --json /tmp/ablation.json
                    python scripts/ablation_report.py --section staleness
                    python scripts/ablation_report.py --list-artifacts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime

#
## ---------------------------------------------------------------------------------------------------
## Code-version timeline (commit dates, from git history). A result produced before a feature landed
## cannot be compared on that feature against a current run.
## ---------------------------------------------------------------------------------------------------
FEATURE_DATES = {
    "balanced_metrics": "2026-10-05",   # b98b8eb: Balanced-count / Occupancy / MAE / eval_threshold
    "empty_gate": "2026-10-05",         # 701912c: staged_count / empty_threshold / predict_counts_staged
    "new_mapping": "2026-10-05",        # 8b84e2f: shared TX-anchored layouts frame
    "layout_free": "2026-10-12",        # d64ce94: layout_free rule + Mass/Position metrics
    "preprocess": "2026-10-12",         # 6a31752: data.preprocess / csi_preprocess.py
    "thatv": "2026-10-12",              # 3f01134: THATV backbone
    "coords_head": "2026-10-13",        # 0a01283: density.head = "coords"
    "cnn_backbone": "2026-10-14",       # 9e64c60: nn.backbone = "cnn"
    "gate_consistent_layoutfree": "2026-10-18",  # 39b2304: gate zeroes peak count + positions
    "sgd": "2026-10-22",                # 461c121: nn.optimizer / nn.momentum
    "coords_decode_fix": "2026-10-23",  # a3a1abd: fix coords decode
}
HEAD_DATE = "2026-10-23"

BACKBONES = ("default", "thatv", "thatv-nostats", "thatv-nopool", "thatv-notokens", "cnn")
HEADS = ("density", "coords")
ROOMS = ("empty_room", "meeting_room", "classroom")
DENSITY_MODELS = ("density_map", "density_map_dem")

#
## Report label -> internal metric key.
DENSITY_LABELS = {
    "Exact-count Accuracy": "exact_count",
    "Balanced-count Accuracy": "balanced_count",
    "Count MAE": "count_mae",
    "Occupancy Accuracy": "occupancy_accuracy",
    "Occupancy F1": "occupancy_f1",
    "Layout-free Mass Accuracy": "mass_accuracy",
    "Layout-free Mass MAE": "mass_mae",
    "Position Error (m)": "position_error_m",
    "Position Miss Rate": "position_miss_rate",
}
AVG_KEYS = {
    "accuracy": "exact_count", "count acc": "exact_count",
    "balanced": "balanced_count",
    "mae": "count_mae",
    "occ acc": "occupancy_accuracy",
    "occ f1": "occupancy_f1",
    "poserr": "position_error_m",
}
OTHER_LABELS = {
    "Avg F1 Score": "f1",
    "Avg Precision": "precision",
    "Avg Recall": "recall",
    "Avg Perfect Prediction %": "ppp",
    "Avg Accuracy": "accuracy",
    "Avg Total Error": "total_error",
}
NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
BLOCK_HEADERS = {"PER_ENV_RESULTS", "SINGLE MODEL RESULTS", "LAYERED MODEL RESULTS",
                 "GROUP-COUNT METRICS", "LOCALIZATION (normalized room units)"}


#
## ---------------------------------------------------------------------------------------------------
## parsing helpers
## ---------------------------------------------------------------------------------------------------
def _f(var_raw):
    try:
        var_value = float(var_raw)
    except (TypeError, ValueError):
        return None
    if var_value != var_value or var_value in (float("inf"), float("-inf")):
        return None
    return var_value


def _sections(var_text):
    """PER_ENV_RESULTS -> {env: [lines]}; empty dict when the report is a single-model block.

    Only a lone `<room>:` (or `layer_N:`) line opens a section, so noise such as the wandb log
    preamble's `wandb:` line never masquerades as an environment.
    """
    var_out, var_cur = {}, None
    for var_line in var_text.splitlines():
        var_match = re.match(r"^(\w+):\s*$", var_line)
        var_token = var_match.group(1) if var_match else None
        if var_token and (var_token in ROOMS or var_token.startswith("layer_")) \
                and var_token not in BLOCK_HEADERS:
            var_cur = var_token
            var_out.setdefault(var_cur, [])
        elif var_cur is not None:
            var_out[var_cur].append(var_line)
    return var_out


def _metrics_from_lines(var_lines):
    var_metrics = {}
    var_text = var_lines if isinstance(var_lines, str) else "\n".join(var_lines)
    for var_label, var_key in DENSITY_LABELS.items():
        var_match = re.search(rf"^\s*{re.escape(var_label)}:\s*{NUM}", var_text, re.M)
        if var_match:
            var_value = _f(var_match.group(1))
            if var_value is not None:
                var_metrics[var_key] = var_value
    return var_metrics


def _merge_avg_lines(var_text, var_env_metrics):
    """Fill position error / count metrics from the per-env 'avg over N repeats' line."""
    for var_match in re.finditer(r"^\[(\w+)\] avg over \d+ repeats:\s*(.*)$", var_text, re.M):
        var_env, var_tail = var_match.group(1), var_match.group(2)
        var_bucket = var_env_metrics.setdefault(var_env, {})
        for var_chunk in var_tail.split("|"):
            var_pair = re.match(rf"\s*(.+?)\s*{NUM}\s*±", var_chunk)
            if not var_pair:
                continue
            var_key = AVG_KEYS.get(var_pair.group(1).strip().lower())
            var_value = _f(var_pair.group(2))
            if var_key and var_value is not None:
                var_bucket.setdefault(var_key, var_value)
    ## coords head: "  [classroom] COUNT Acc X - Bal Y - MAE Z - PosErr W m"
    for var_match in re.finditer(
            rf"^\s*\[(\w+)\] COUNT Acc {NUM} - Bal {NUM} - MAE {NUM} - PosErr {NUM} m", var_text, re.M):
        var_bucket = var_env_metrics.setdefault(var_match.group(1), {})
        var_bucket.setdefault("exact_count", _f(var_match.group(2)))
        var_bucket.setdefault("balanced_count", _f(var_match.group(3)))
        var_bucket.setdefault("count_mae", _f(var_match.group(4)))
        var_bucket.setdefault("position_error_m", _f(var_match.group(5)))
    ## density head single-room: "  [classroom] COUNT: Acc X - Bal Y - MAE Z - Occupancy Acc W - Occupancy F1 V"
    for var_match in re.finditer(
            rf"^\s*\[(\w+)\] COUNT: Acc {NUM} - Bal {NUM} - MAE {NUM} - Occupancy Acc {NUM} - Occupancy F1 {NUM}",
            var_text, re.M):
        var_bucket = var_env_metrics.setdefault(var_match.group(1), {})
        for var_key, var_group in (("exact_count", 2), ("balanced_count", 3), ("count_mae", 4),
                                   ("occupancy_accuracy", 5), ("occupancy_f1", 6)):
            var_bucket.setdefault(var_key, _f(var_match.group(var_group)))


def parse_density_report(var_text):
    """{env: {metric: value}} from a density/coords report block (SINGLE or PER_ENV)."""
    var_env_metrics = {}
    var_sections = _sections(var_text)
    if var_sections:
        for var_env, var_lines in var_sections.items():
            var_env_metrics[var_env] = _metrics_from_lines(var_lines)
    else:
        var_env_metrics[None] = _metrics_from_lines(var_text)
    _merge_avg_lines(var_text, var_env_metrics)
    return {var_env: var_metrics for var_env, var_metrics in var_env_metrics.items() if var_metrics}


def parse_other_report(var_text):
    """Non-density reports (AMAR_* layered / single) -> {layer: {f1, precision, ...}}."""
    var_out = {}
    var_sections = _sections(var_text)
    if not var_sections:
        var_metrics = {}
        for var_label, var_key in OTHER_LABELS.items():
            var_match = re.search(rf"^\s*{re.escape(var_label)}:\s*{NUM}", var_text, re.M)
            if var_match:
                var_value = _f(var_match.group(1))
                if var_value is not None:
                    var_metrics[var_key] = var_value
        if var_metrics:
            var_out[None] = var_metrics
        return var_out
    for var_layer, var_lines in var_sections.items():
        var_metrics = {}
        for var_label, var_key in OTHER_LABELS.items():
            var_match = re.search(rf"^\s*{re.escape(var_label)}:\s*{NUM}", "\n".join(var_lines), re.M)
            if var_match:
                var_value = _f(var_match.group(1))
                if var_value is not None:
                    var_metrics[var_key] = var_value
        if var_metrics:
            var_out[var_layer] = var_metrics
    ## a layered report exposes the last decoder layer as the trained output
    var_layer_keys = sorted((var_key for var_key in var_out if var_key and var_key.startswith("layer_")),
                            key=lambda var_key: int(var_key.split("_")[-1]))
    if var_layer_keys:
        return {"__final__": var_out[var_layer_keys[-1]]}
    return var_out


def parse_few_shot_report(var_text):
    """H-WILD -> WiMANS few-shot block -> {env: {student_mde, teacher_mde, const_mde}}."""
    var_out = {}
    for var_match in re.finditer(
            rf"^(\w+):\s*\n\s*Student\s+MDE\s+{NUM}[^\n]*\n\s*Teacher\s+MDE\s+{NUM}[^\n]*\n"
            rf"\s*Constant\s+MDE\s+{NUM}", var_text, re.M):
        var_out[var_match.group(1)] = {"student_mde": _f(var_match.group(2)),
                                       "teacher_mde": _f(var_match.group(3)),
                                       "const_mde": _f(var_match.group(4))}
    return var_out


def parse_localization_report(var_text):
    """H-WILD / unified single-target block -> {env: {mean_error_m, const_error_m}}."""
    var_out = {}
    for var_match in re.finditer(
            rf"^(\w+):\s*\n\s*Mean Distance Error:\s*{NUM}[^\n]*\n(?:[^\n]*\n)*?"
            rf"\s*Constant-baseline:\s*{NUM}", var_text, re.M):
        var_out[var_match.group(1)] = {"mean_error_m": _f(var_match.group(2)),
                                       "const_error_m": _f(var_match.group(3))}
    return var_out


def parse_gate_diagnosis(var_text):
    """EMPTY GATE block of a diagnose log -> separability stats."""
    var_match = re.search(
        rf"EMPTY GATE:\s*(\d+) empty / (\d+) occupied frames\s*\n"
        rf"\s*map max empty\s*{NUM}\s*\+-\s*{NUM}\s*\n"
        rf"\s*map max occupied\s*{NUM}\s*\+-\s*{NUM}\s*\n"
        rf"\s*best gate threshold={NUM}\s*balanced empty/occupied acc={NUM}", var_text)
    if not var_match:
        return None
    var_dep = re.search(rf"shuffled\s+abs={NUM}\s+rel={NUM}", var_text)
    return {"empty_frames": int(var_match.group(1)), "occupied_frames": int(var_match.group(2)),
            "max_empty": _f(var_match.group(3)), "max_occupied": _f(var_match.group(5)),
            "best_threshold": _f(var_match.group(7)), "best_balanced": _f(var_match.group(8)),
            "shuffled_rel": _f(var_dep.group(2)) if var_dep else None}


def schema_of(var_env_metrics):
    """Ordered schema id from the density metric keys present."""
    var_keys = set()
    for var_metrics in var_env_metrics.values():
        var_keys |= set(var_metrics)
    if "mass_accuracy" in var_keys:
        return "S3"                       # layout-free suite (>= 2026-10-12)
    if "balanced_count" in var_keys:
        return "S2c" if var_keys == {"exact_count", "balanced_count", "count_mae"} else "S2"
    if "exact_count" in var_keys:
        return "S1"
    return "?"


#
## ---------------------------------------------------------------------------------------------------
## artifact record
## ---------------------------------------------------------------------------------------------------
@dataclass
class Record:
    path: str
    protocol: str
    model: str | None = None
    archived: bool = False
    run_time: str | None = None
    head: str | None = None
    backbone: str | None = None
    encoder_norm: str | None = None
    count_rule: str | None = None
    train_envs: list = field(default_factory=list)
    test_envs: list = field(default_factory=list)
    holdout: str | None = None
    repeat: int | None = None
    epochs: int | None = None
    schema: str = "?"
    metrics: dict = field(default_factory=dict)
    gate: dict | None = None
    version: str = "?"
    stale: list = field(default_factory=list)


def _stamp(var_path, var_name):
    var_match = re.search(r"(\d{8})_(\d{6})", var_name) or re.search(r"(\d{8})_(\d{4})", var_name)
    if var_match:
        var_raw = var_match.group(1) + var_match.group(2)
        var_fmt = "%Y%m%d%H%M%S" if len(var_match.group(2)) == 6 else "%Y%m%d%H%M"
        try:
            return datetime.strptime(var_raw, var_fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(var_path)).strftime("%Y-%m-%d")
    except OSError:
        return None


def _cross_domain_stem(var_name):
    var_match = re.search(r"train-([\w+]+)_test-([\w+]+)_r(\d+)_e(\d+)", var_name)
    if not var_match:
        return [], [], None, None
    return (var_match.group(1).split("+"), var_match.group(2).split("+"),
            int(var_match.group(3)), int(var_match.group(4)))


def build_record(var_path, var_rel):
    var_parts = var_rel.split(os.sep)
    var_name = os.path.basename(var_rel)
    var_rec = Record(path=var_rel, protocol="other")
    var_rec.run_time = _stamp(var_path, var_name)

    if var_parts[0] == "matrix" and len(var_parts) >= 6:
        _, var_rec.model, var_rec.head, var_rec.backbone, var_rec.encoder_norm, var_env = var_parts[:6]
        var_rec.protocol = "matrix"
        var_rec.test_envs = [var_env]
    elif var_parts[0] == "lor_matrix" and len(var_parts) >= 6:
        ## two on-disk layouts: with a head token (density/coords) and the older one without it
        var_rec.model = var_parts[1]
        var_idx = 2
        if var_parts[var_idx] in HEADS:
            var_rec.head = var_parts[var_idx]
            var_idx += 1
        if var_parts[var_idx] in BACKBONES:
            var_rec.backbone = var_parts[var_idx]
            var_idx += 1
        if var_idx < len(var_parts) and var_parts[var_idx] in ("batch", "group", "layer"):
            var_rec.encoder_norm = var_parts[var_idx]
        var_hold = next((var_part for var_part in var_parts if var_part.startswith("holdout_")), None)
        var_rec.protocol = "lor_matrix"
        if var_hold:
            var_rec.holdout = var_hold.replace("holdout_", "")
            var_rec.test_envs = [var_rec.holdout]
            var_rec.train_envs = [var_room for var_room in ROOMS if var_room != var_rec.holdout]
    elif var_parts[0] == "density_map" and len(var_parts) >= 3:
        var_rec.protocol = "density_map"
        var_rec.model = "density_map"
        if var_parts[1] in BACKBONES:
            var_rec.backbone, var_env = var_parts[1], var_parts[2]
        else:
            var_env = var_parts[1]
        var_rec.test_envs = [var_env] if var_env in ROOMS else []
    elif var_parts[0] == "cross_domain":
        var_rec.protocol = "cross_domain"
        var_rec.train_envs, var_rec.test_envs, var_rec.repeat, var_rec.epochs = _cross_domain_stem(var_name)
    elif var_parts[0] == "single":
        var_rec.protocol = "single"
        var_rec.model = var_name.split("_")[0]
        var_match = re.search(r"env-(\w+?)_n(?:all|\d+)", var_name)
        var_rec.test_envs = [var_match.group(1)] if var_match else []
    elif var_parts[:2] == ["sweep", "few_shot_hwild_wimans"]:
        var_rec.protocol = "sweep_few_shot"
        var_match = re.search(r"kd([\d.]+)_fs([\d.]+)_init(\w+)_std(\w+)", var_rel)
        if var_match:
            var_rec.count_rule = f"kd={var_match.group(1)} init={var_match.group(3)} std={var_match.group(4)}"
    elif var_parts[0] == "few_shot_hwild_to_wimans":
        var_rec.protocol = "few_shot"
        var_match = re.search(r"fs([\d.]+)_kd([\d.]+)_r(\d+)_(window|none)_\d", var_name)
        if var_match:
            var_rec.count_rule = f"kd={var_match.group(2)} std={var_match.group(4)}"
    elif var_parts[0] == "hwild_to_wimans":
        var_rec.protocol = "hwild_to_wimans"
        var_match = re.search(r"_(window|none)_\d{8}", var_name)
        var_rec.count_rule = f"std={var_match.group(1)}" if var_match else None
    elif var_parts[0] == "diagnose":
        var_rec.protocol = "diagnose"
    elif var_parts[0] == "cd":
        var_rec.protocol = "cd_legacy"
        var_match = re.search(r"train-([\w+]+)_test-([\w+]+)", var_name)
        if var_match:
            var_rec.train_envs, var_rec.test_envs = var_match.group(1).split("+"), var_match.group(2).split("+")
    return var_rec


def classify(var_rec, var_text):
    var_header = re.search(r"Model:\s*(\w+)", var_text)
    if var_header:
        var_rec.model = var_header.group(1)

    if var_rec.protocol in ("few_shot", "sweep_few_shot", "hwild_to_wimans"):
        var_few = parse_few_shot_report(var_text)
        var_rec.metrics = var_few or parse_localization_report(var_text)
        var_rec.schema = "S8" if var_few else "S9"
        var_rec.version = FEATURE_DATES["sgd"]
        return var_rec
    if var_rec.protocol == "diagnose":
        var_rec.schema = "S10"
        var_rec.gate = parse_gate_diagnosis(var_text)
        var_rec.version = FEATURE_DATES["preprocess"]
        return var_rec

    var_rec.metrics = parse_density_report(var_text)
    if var_rec.metrics:
        var_rec.schema = schema_of(var_rec.metrics)
        ## the coords head is the only density-family model that prints a per-epoch "PosErr"
        if var_rec.head is None and re.search(r"\bPosErr\b", var_text):
            var_rec.head = "coords"
        var_has_mass = any("mass_accuracy" in var_metrics for var_metrics in var_rec.metrics.values())
        if var_rec.count_rule is None:
            if var_has_mass:
                var_rec.count_rule = "layout_free"
            elif var_rec.schema in ("S1", "S2") and var_rec.head == "density":
                var_rec.count_rule = "template"
        ## version bracket = newest feature the artifact demonstrably used
        var_dates = [FEATURE_DATES["balanced_metrics"]] if var_rec.schema in ("S1", "S2", "S2c") else []
        if var_has_mass:
            var_dates.append(FEATURE_DATES["layout_free"])
        if var_rec.head == "coords":
            var_dates.append(FEATURE_DATES["coords_head"])
        if var_rec.backbone and var_rec.backbone.startswith("thatv"):
            var_dates.append(FEATURE_DATES["thatv"])
        if var_rec.backbone == "cnn":
            var_dates.append(FEATURE_DATES["cnn_backbone"])
        var_rec.version = max(var_dates) if var_dates else "< 2026-10-05"
        ## staleness relative to HEAD
        var_run = var_rec.run_time or "9999-99-99"
        if var_rec.model in DENSITY_MODELS:
            if var_rec.head == "coords":
                if var_run < FEATURE_DATES["coords_decode_fix"]:
                    var_rec.stale.append(
                        f"coords head before the coords-decode fix ({FEATURE_DATES['coords_decode_fix']})")
            elif var_rec.schema in ("S1", "S2", "S2c"):
                if var_run < FEATURE_DATES["layout_free"]:
                    var_rec.stale.append(
                        f"report lacks the layout-free metric block (pre-{FEATURE_DATES['layout_free']})")
            if var_rec.count_rule == "layout_free" and var_run < FEATURE_DATES["gate_consistent_layoutfree"]:
                var_rec.stale.append(
                    f"layout-free before gate-consistent counts ({FEATURE_DATES['gate_consistent_layoutfree']})")
        return var_rec

    var_rec.metrics = parse_other_report(var_text)
    var_rec.schema = "other"
    var_rec.version = "< 2026-10-05"
    return var_rec


#
## ---------------------------------------------------------------------------------------------------
## discovery
## ---------------------------------------------------------------------------------------------------
def collect(var_output_dir):
    """Parse every report under one or more roots (e.g. the live output/ and old_results/output/)."""
    var_roots = [var_output_dir] if isinstance(var_output_dir, str) else list(var_output_dir)
    var_records = []
    for var_root in var_roots:
        for var_path in sorted(glob.glob(os.path.join(var_root, "**", "*"), recursive=True)):
            if not os.path.isfile(var_path) or not var_path.endswith((".txt", ".log")):
                continue
            var_rel = os.path.relpath(var_path, var_root)
            if var_rel.startswith("wandb"):
                continue
            try:
                with open(var_path, "r", errors="ignore") as var_file:
                    var_text = var_file.read()
            except OSError:
                continue
            if "EXPERIMENT RESULTS" not in var_text and "avg over" not in var_text \
                    and "Student   MDE" not in var_text and "Mean Distance Error:" not in var_text \
                    and "INPUT DEPENDENCE" not in var_text:
                continue
            var_rec = build_record(var_path, var_rel)
            var_rec.archived = os.path.abspath(var_root) != os.path.abspath("output")
            classify(var_rec, var_text)
            if var_rec.metrics or var_rec.gate:
                var_records.append(var_rec)
    return var_records


#
## ---------------------------------------------------------------------------------------------------
## analyzers
## ---------------------------------------------------------------------------------------------------
def _mean(var_values):
    var_values = [var_value for var_value in var_values if var_value is not None]
    return sum(var_values) / len(var_values) if var_values else None


def _fmt(var_value, var_digits=4):
    return "-" if var_value is None else f"{var_value:.{var_digits}f}"


def _disp(var_rec):
    """Artifact path for display, tagged with the tree it lives in."""
    var_root = "old_results/output" if var_rec.archived else "output"
    return f"{var_root}/{var_rec.path}"


def _fold_of(var_rec):
    return var_rec.holdout or (var_rec.test_envs[0] if var_rec.test_envs else None)


def analyze_calibration(var_output_dir, var_records, var_out):
    """Q1: is the staged (two-stage) count good, and which calibration is best?"""
    var_rows = defaultdict(list)
    for var_index in glob.glob(os.path.join(var_output_dir, "sweep", "density_map_*", "thresholds*", "index.tsv")):
        var_match = re.search(r"density_map_(\w+)[/\\](thresholds\w*)", var_index)
        var_room, var_variant = (var_match.group(1), var_match.group(2)) if var_match else ("?", "?")
        with open(var_index, errors="ignore") as var_file:
            for var_line in var_file:
                var_cells = var_line.rstrip("\n").split("\t")
                if len(var_cells) != 8:
                    continue
                var_rule, var_knob, var_value, var_metric, var_score, _var_tie, var_tie_score, _ = var_cells
                var_rows[(var_room, var_rule, var_knob, var_variant)].append({
                    "value": _f(var_value), "metric": var_metric, "score": _f(var_score), "tie": _f(var_tie_score)})

    var_out.append("## Q1 - staged count and its calibration\n")
    var_out.append("Source: output/sweep/density_map_<room>/thresholds{,_bis}/index.tsv, scored on the "
                   "VALIDATION split (never the reported one).\n")
    if not var_rows:
        var_out.append("  no threshold-sweep artifacts found\n")
    else:
        var_out.append(f"{'room':12s} {'rule':12s} {'knob':6s} {'variant':14s} {'lowest':>8s} "
                       f"{'best':>8s} {'delta':>8s} {'best value':>10s}  metric")
        var_out.append("-" * 96)
        for (var_room, var_rule, var_knob, var_variant), var_cells in sorted(var_rows.items()):
            var_valid = [var_cell for var_cell in var_cells if var_cell["score"] is not None]
            if not var_valid:
                continue
            var_best = sorted(var_valid, key=lambda var_cell: (-var_cell["score"],
                                                              var_cell["tie"] if var_cell["tie"] is not None else 9e9))[0]
            var_low = sorted(var_valid, key=lambda var_cell: abs(var_cell["value"] if var_cell["value"] is not None else 9e9))[0]
            var_delta = (var_best["score"] - var_low["score"]) if var_best["score"] is not None and var_low["score"] is not None else None
            var_out.append(f"{var_room:12s} {var_rule:12s} {var_knob:6s} {var_variant:14s} "
                           f"{_fmt(var_low['score']):>8s} {_fmt(var_best['score']):>8s} "
                           f"{_fmt(var_delta):>8s} {_fmt(var_best['value'], 3):>10s}  {var_best['metric']}")
        var_out.append("\n  'lowest' = the cell with the smallest knob value (empty=0.0 => the empty gate never "
                       "fires; eval=0.0 => the degenerate all-occupied threshold).")
        var_out.append("  The empty knob is scored by valid_avg_balanced_accuracy (both rules); the eval knob by "
                       "valid_avg_balanced_accuracy (template) / valid_avg_occupancy_f1 (layout_free).")

    var_out.append("\n  Gate separability, from output/diagnose/<room>/*.log (fitted on the source room, an upper bound):")
    var_out.append(f"  {'room':13s} {'max_empty':>10s} {'max_occupied':>13s} {'best thr':>9s} {'balanced':>9s} "
                   f"{'shuffled rel':>13s}")
    for var_rec in sorted(var_records, key=lambda var_rec: var_rec.path):
        if var_rec.protocol != "diagnose" or not var_rec.gate:
            continue
        var_gate = var_rec.gate
        var_room = os.path.basename(os.path.dirname(var_rec.path))
        var_out.append(f"  {var_room:13s} {_fmt(var_gate['max_empty'],3):>10s} {_fmt(var_gate['max_occupied'],3):>13s} "
                       f"{_fmt(var_gate['best_threshold'],3):>9s} {_fmt(var_gate['best_balanced'],3):>9s} "
                       f"{_fmt(var_gate['shuffled_rel'],3):>13s}")
    var_out.append("")


def analyze_heads(var_records, var_out):
    """Q3: coords head vs density head, on matched backbone/encoder/fold."""
    var_out.append("## Q3 - coords head vs density head\n")
    var_keyed = defaultdict(lambda: defaultdict(list))
    for var_rec in var_records:
        if var_rec.protocol not in ("lor_matrix", "matrix") or var_rec.head not in HEADS:
            continue
        var_fold = _fold_of(var_rec) or "?"
        var_metrics = var_rec.metrics.get(var_fold) or next(iter(var_rec.metrics.values()), {})
        var_keyed[(var_rec.protocol, var_rec.backbone, var_rec.encoder_norm, var_fold)][var_rec.head].append(var_metrics)
    var_out.append(f"{'protocol':11s} {'backbone':14s} {'enc':6s} {'fold':13s} "
                   f"{'density PosErr':>14s} {'coords PosErr':>13s} {'delta':>8s} {'d/c exact-count':>16s}")
    var_out.append("-" * 110)
    for var_key, var_heads in sorted(var_keyed.items()):
        if "density" not in var_heads or "coords" not in var_heads:
            continue
        var_protocol, var_backbone, var_encoder, var_fold = var_key
        var_dp = _mean([m.get("position_error_m") for m in var_heads["density"]])
        var_cp = _mean([m.get("position_error_m") for m in var_heads["coords"]])
        var_de = _mean([m.get("exact_count") for m in var_heads["density"]])
        var_ce = _mean([m.get("exact_count") for m in var_heads["coords"]])
        var_delta = (var_cp - var_dp) if var_dp is not None and var_cp is not None else None
        var_out.append(f"{var_protocol:11s} {str(var_backbone):14s} {str(var_encoder):6s} {var_fold:13s} "
                       f"{_fmt(var_dp):>8s}({len(var_heads['density']):>2d}) {_fmt(var_cp):>8s}"
                       f"({len(var_heads['coords']):>2d}) {_fmt(var_delta):>8s} "
                       f"{_fmt(var_de,3)+' / '+_fmt(var_ce,3):>16s}")
    var_out.append("")
    var_dp_all = [_mean([m.get("position_error_m") for m in var_heads["density"]])
                  for var_heads in var_keyed.values() if "density" in var_heads]
    var_cp_all = [_mean([m.get("position_error_m") for m in var_heads["coords"]])
                  for var_heads in var_keyed.values() if "coords" in var_heads]
    var_out.append(f"  mean Position Error (m): density {_fmt(_mean(var_dp_all))}   coords {_fmt(_mean(var_cp_all))} "
                   f"(lower is better; negative delta = coords better)")
    var_out.append("  Note: in the matrix runs the density head's numbers are identical across encoder_norm "
                   "batch/group/layer for a given backbone (nn.encoder_norm was a no-op there), so those rows "
                   "are one measurement repeated, not three independent ones.")
    var_out.append("  Source: matrix/ and lor_matrix/ runs, now archived under old_results/output/ because they "
                   "predate the 2026-10-23 coords-decode fix and the 2026-10-18 gate-consistent counts.\n")


def analyze_backbones(var_records, var_out):
    """Q4: which backbone transfers best?"""
    var_out.append("## Q4 - backbones\n")
    var_groups = defaultdict(list)
    for var_rec in var_records:
        if var_rec.protocol not in ("lor_matrix", "matrix", "density_map") or not var_rec.backbone:
            continue
        var_fold = _fold_of(var_rec) or "?"
        var_metrics = var_rec.metrics.get(var_fold) or next(iter(var_rec.metrics.values()), {})
        var_groups[(var_rec.protocol, var_rec.model, var_rec.head or "?", var_rec.backbone)].append(var_metrics)
    var_out.append(f"{'protocol':11s} {'model':13s} {'head':8s} {'backbone':14s} {'n':>3s} "
                   f"{'PosErr m':>9s} {'massAcc':>8s} {'exactCnt':>9s} {'countMAE':>9s} {'f1':>7s}")
    var_out.append("-" * 102)
    for var_key, var_metrics in sorted(var_groups.items()):
        var_protocol, var_model, var_head, var_backbone = var_key
        var_out.append(f"{var_protocol:11s} {str(var_model):13s} {str(var_head):8s} {var_backbone:14s} "
                       f"{len(var_metrics):>3d} "
                       f"{_fmt(_mean([m.get('position_error_m') for m in var_metrics])):>9s} "
                       f"{_fmt(_mean([m.get('mass_accuracy') for m in var_metrics])):>8s} "
                       f"{_fmt(_mean([m.get('exact_count') for m in var_metrics])):>9s} "
                       f"{_fmt(_mean([m.get('count_mae') for m in var_metrics])):>9s} "
                       f"{_fmt(_mean([m.get('f1') for m in var_metrics]), 3):>7s}")
    var_out.append("  Note: matrix/, lor_matrix/ and output/density_map/ backbone runs are archived (stale); "
                   "the cross-domain rows (lor_matrix) are the only cross-room backbone evidence.\n")


def analyze_optimizer(var_records, var_out):
    """Q5: is SGD better?"""
    var_out.append("## Q5 - SGD optimizer\n")
    var_hits = [var_rec for var_rec in var_records
                if "sgd" in var_rec.path.lower() or "momentum" in var_rec.path.lower()
                or (var_rec.count_rule and "sgd" in str(var_rec.count_rule).lower())]
    if var_hits:
        for var_rec in var_hits:
            var_out.append(f"  {_disp(var_rec)}")
    else:
        var_out.append("  No artifact exercises nn.optimizer. No launcher passes --hp nn.optimizer=sgd, and no "
                       "report/log records the optimizer, so Adam-vs-SGD cannot be answered from disk.\n")


def analyze_preprocessing(var_records, var_out):
    """Q6: do the preprocessing steps help?"""
    var_out.append("## Q6 - preprocessing\n")
    var_few = {}
    for var_rec in var_records:
        if var_rec.protocol not in ("few_shot", "sweep_few_shot") or not var_rec.count_rule:
            continue
        if not var_rec.path.endswith(".txt"):
            continue   ## one record per run: the .txt, not the duplicate run.log
        var_few[var_rec.count_rule] = var_rec.metrics
    for var_config, var_metrics in sorted(var_few.items()):
        var_out.append(f"  H-WILD->WiMANS few-shot  {var_config:30s} student MDE "
                       f"{_fmt(_mean([m.get('student_mde') for m in var_metrics.values()]),3)} m")
    var_std = {re.search(r"std=(\w+)", var_config).group(1) for var_config in var_few if "std=" in var_config}
    var_out.append(f"\n  standardize variants present: {sorted(var_std) or 'none'}")
    var_out.append("  No AMAR_CSI_PREPROCESS / data.preprocess training artifact exists: static_removal, "
                   "temporal_zscore and per_room_zscore have never been run for training. Only output/diagnose/ "
                   "logs exist (preprocess=none), which measure input dependence, not accuracy.\n")


def analyze_mapping(var_records, var_out):
    """Q2: did the coordinate-mapping change help?"""
    var_out.append("## Q2 - coordinate mapping\n")
    var_groups = defaultdict(list)
    for var_rec in var_records:
        if var_rec.protocol not in ("cross_domain", "cd_legacy", "lor_matrix", "single"):
            continue
        var_fold = _fold_of(var_rec)
        if not var_fold:
            continue
        var_metrics = var_rec.metrics.get(var_fold) or next(iter(var_rec.metrics.values()), {})
        var_era = ("new-mapping + layout-free" if var_rec.schema == "S3"
                   else "new-mapping, pre-layout-free"
                   if var_rec.run_time and var_rec.run_time >= FEATURE_DATES["new_mapping"]
                   else "pre-new-mapping")
        var_groups[(var_fold, var_era)].append(var_metrics)
    var_out.append(f"{'fold':13s} {'era':28s} {'n':>3s} {'PosErr m':>9s} {'exactCnt':>9s} {'countMAE':>9s}")
    var_out.append("-" * 78)
    for var_key, var_metrics in sorted(var_groups.items()):
        var_fold, var_era = var_key
        var_out.append(f"{var_fold:13s} {var_era:28s} {len(var_metrics):>3d} "
                       f"{_fmt(_mean([m.get('position_error_m') for m in var_metrics])):>9s} "
                       f"{_fmt(_mean([m.get('exact_count') for m in var_metrics])):>9s} "
                       f"{_fmt(_mean([m.get('count_mae') for m in var_metrics])):>9s}")
    var_out.append("\n  No launcher overrides preset['layouts'], so there is no controlled A/B on the mapping. "
                   "Position Error (m) only exists from the layout-free era (>= 2026-10-12), so the "
                   "pre-new-mapping era cannot be scored in meters; any across-era difference is confounded with "
                   "the metric-suite change.\n")


def analyze_staleness(var_records, var_out):
    """Which artifacts are from a different code version and must not be compared to current runs?"""
    var_out.append("## Outdated artifacts (different code version than HEAD)\n")
    var_out.append(f"HEAD = {HEAD_DATE}. An artifact is stale for an axis when it predates the feature that "
                   "axis depends on.\n")
    var_stale = [var_rec for var_rec in var_records if var_rec.stale]
    var_by_reason = defaultdict(list)
    for var_rec in var_stale:
        for var_reason in var_rec.stale:
            var_by_reason[var_reason].append(var_rec)
    for var_reason, var_recs in sorted(var_by_reason.items(), key=lambda var_item: -len(var_item[1])):
        var_out.append(f"  [{len(var_recs):3d}] {var_reason}")
        for var_rec in var_recs[:4]:
            var_out.append(f"        e.g. {_disp(var_rec)}")
        if len(var_recs) > 4:
            var_out.append(f"        ... {len(var_recs) - 4} more")
    var_live_stale = [var_rec for var_rec in var_stale if not var_rec.archived]
    var_out.append(f"\n  live output/: {sum(1 for r in var_records if not r.archived)} artifacts, "
                   f"{len(var_live_stale)} stale")
    var_out.append(f"  old_results/output/: {sum(1 for r in var_records if r.archived)} artifacts "
                   f"(all archived as outdated)")
    var_out.append("\n  Version bracket by schema (newest feature the report demonstrably used):")
    var_by_version = defaultdict(int)
    for var_rec in var_records:
        var_by_version[var_rec.version] += 1
    for var_version, var_count in sorted(var_by_version.items()):
        var_out.append(f"    >= {var_version}: {var_count}")
    var_out.append("")


#
## ---------------------------------------------------------------------------------------------------
def main():
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--output-dir", default="output", type=str)
    var_args.add_argument("--json", default=None, type=str, help="dump the parsed records as JSON")
    var_args.add_argument("--section", default=None, type=str,
                          choices=("calibration", "heads", "backbones", "optimizer", "preprocessing",
                                   "mapping", "staleness"),
                          help="print only one section")
    var_args.add_argument("--list-artifacts", action="store_true", help="print one line per parsed artifact")
    var_args = var_args.parse_args()

    var_roots = [var_args.output_dir]
    var_archive = os.path.join("old_results", "output")
    if os.path.isdir(var_archive) and os.path.abspath(var_archive) not in [os.path.abspath(var_roots[0])]:
        var_roots.append(var_archive)
    var_records = collect(var_roots)
    var_sections = {
        "calibration": lambda out: analyze_calibration(var_args.output_dir, var_records, out),
        "heads": lambda out: analyze_heads(var_records, out),
        "backbones": lambda out: analyze_backbones(var_records, out),
        "optimizer": lambda out: analyze_optimizer(var_records, out),
        "preprocessing": lambda out: analyze_preprocessing(var_records, out),
        "mapping": lambda out: analyze_mapping(var_records, out),
        "staleness": lambda out: analyze_staleness(var_records, out),
    }
    var_out = ["=" * 96,
               f"AMAR ABLATION REPORT - {len(var_records)} result artifacts under "
               f"{' + '.join(var_roots)}",
               "=" * 96 + "\n"]
    if var_args.list_artifacts:
        for var_rec in sorted(var_records, key=lambda var_rec: var_rec.path):
            var_out.append(f"{var_rec.schema:5s} {str(var_rec.version):12s} {var_rec.protocol:14s} "
                           f"{str(var_rec.model):13s} {str(var_rec.head):8s} {str(var_rec.backbone):14s} "
                           f"{str(var_rec.encoder_norm):6s} {str(var_rec.count_rule):12s} {_disp(var_rec)}")
        var_out.append("")
    else:
        for var_name in ([var_args.section] if var_args.section else list(var_sections)):
            var_sections[var_name](var_out)
    print("\n".join(var_out))

    if var_args.json:
        with open(var_args.json, "w") as var_file:
            json.dump([asdict(var_rec) for var_rec in var_records], var_file, indent=2)
        print(f"\nrecords written to {var_args.json}")


if __name__ == "__main__":
    main()
