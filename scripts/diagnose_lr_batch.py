"""
[file]          diagnose_lr_batch.py
[description]   Sweep the localizer's learning rate x batch size and write a diagnosis file, to answer
                "will a smaller lr / larger batch improve my runs?" without hand-editing the presets.

                Three protocols (--protocol):
                  diag          one room, the same-room diagnostic (scripts/test_scripts/
                                diagnose_localizer_fit.py): trains and validates on one room's split and
                                reports the best validation error. Cheap, isolates the optimiser.
                  cross_domain  the real cross-domain runner (scripts/run_cross_domain.py with the
                                ResNet1D localizer): trains on --train_envs, tests on every other room.
                  few_shot      the real few-shot runner (scripts/run_few_shot_multi_rooms.py):
                                teacher on --train_rooms, student on a few-shot slice of
                                --support_rooms, scored per --test_rooms.

                The diag protocol reports the BEST validation error (not the last step), which is the
                number to read: a config is better when its best error is lower, and it has "reached the
                ceiling" when it is at or below the room's nearest-centroid baseline.

                The two real protocols report the per-room mean distance error (MDE) and its mean over
                the scored rooms, next to the constant baseline that the same run prints, so the sweep
                answers "does a different lr/batch lower the MDE I actually report, without crossing the
                constant baseline".

                Every cell is one fresh process, so the fixed seeds make cells comparable.

                Examples:
                    python scripts/diagnose_lr_batch.py
                    python scripts/diagnose_lr_batch.py --protocol cross_domain \
                        --train_envs "meeting_room, empty_room, anywhere" --epochs 70 --repeat 3
                    python scripts/diagnose_lr_batch.py --protocol few_shot \
                        --train_rooms "meeting_room, empty_room, anywhere" \
                        --test_rooms "classroom, meeting_room_synth, empty_room_synth, classroom_synth" \
                        --support_rooms classroom
"""
#
##

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

#
##
HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(HERE, "test_scripts", "diagnose_localizer_fit.py")
CROSS_DOMAIN = os.path.join(HERE, "run_cross_domain.py")
FEW_SHOT = os.path.join(HERE, "run_few_shot_multi_rooms.py")

CONSTANT_RE = re.compile(r"constant baseline\s*:\s*([0-9.]+) m")
CENTROID_RE = re.compile(r"centroid baseline\s*:\s*([0-9.]+) m")
BEST_RE = re.compile(r"valid:\s*best\s*([0-9.]+) m @ step (\d+)\s*\|\s*last\s*([0-9.]+) m"
                     r"(?:\s*\|\s*weight-EMA\s*([0-9.]+) m)?")
ACC_CENTROID_RE = re.compile(r"centroid\s+\(data separability\): per-location accuracy ([0-9.]+)")
ACC_MODEL_RE = re.compile(r"\(trained model\): per-location accuracy ([0-9.]+)")

## cross-domain report (src.models.room_localization.format_localization_result)
CD_HEADER_RE = re.compile(r"^([a-z][a-z_0-9]*):$")
CD_MDE_RE = re.compile(r"Mean Distance Error:\s*([0-9.]+)")
CD_CONST_RE = re.compile(r"Constant-baseline:\s*([0-9.]+)")
## few-shot report (scripts.run_few_shot_multi_rooms.format_result)
FS_HEADER_RE = re.compile(r"^([A-Z][A-Z_0-9]*):$")
FS_STUDENT_RE = re.compile(r"Student MDE:\s*([0-9.]+)")
FS_TEACHER_RE = re.compile(r"Teacher \(frozen, no fine-tuning\) MDE:\s*([0-9.]+)")
FS_CONST_RE = re.compile(r"Constant\(training\) MDE:\s*([0-9.]+)")
SKIP_HEADERS = {"PER_ENV_RESULTS"}


def parse_args():
    """
    [description]
    : command line arguments for the sweep
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    var_args.add_argument("--protocol", default="diag", choices=("diag", "cross_domain", "few_shot"),
                          help="which protocol the sweep measures (see the module docstring)")
    var_args.add_argument("--lrs", default="1e-3 3e-4", type=str,
                          help="space-separated learning rates to sweep")
    var_args.add_argument("--batches", default="32 128", type=str,
                          help="space-separated batch sizes to sweep")
    var_args.add_argument("--backbone", default="resnet1d", choices=("resnet1d", "that"))
    var_args.add_argument("--baseline_lr", default="1e-3", type=str,
                          help="the current lr, for the 'did the sweep help?' comparison")
    var_args.add_argument("--baseline_batch", default="32", type=int,
                          help="the current batch size, for the 'did the sweep help?' comparison")
    var_args.add_argument("--out", default=None, type=str,
                          help="diagnosis file (default output/diagnose/<protocol>_<timestamp>.txt; a "
                               "matching .csv is written beside it)")
    var_args.add_argument("--hp", action="append", default=None, type=str, metavar="KEY=VALUE",
                          help="extra --hp override forwarded to every cell (repeatable)")
    var_args.add_argument("--no_synth", action="store_true",
                          help="do not force preset['data']['synth_wimans']['enabled']=true for the "
                               "real protocols")
    #
    ## diag protocol
    var_args.add_argument("--rooms", default="empty_room_synth meeting_room", type=str,
                          help="diag: space-separated rooms (keys of wimans_room_names())")
    var_args.add_argument("--steps", default=4000, type=int, help="diag: gradient steps per cell")
    #
    ## cross_domain / few_shot protocols
    var_args.add_argument("--epochs", default=70, type=int,
                          help="cross_domain: preset['hwild']['epoch']")
    var_args.add_argument("--repeat", default=3, type=int, help="real protocols: repeats per cell")
    var_args.add_argument("--train_envs", default="meeting_room, empty_room, anywhere", type=str,
                          help="cross_domain: training rooms (comma-separated)")
    var_args.add_argument("--train_rooms", default="meeting_room, empty_room, anywhere", type=str,
                          help="few_shot: teacher rooms (comma-separated)")
    var_args.add_argument("--test_rooms",
                          default="classroom, meeting_room_synth, empty_room_synth, classroom_synth",
                          type=str, help="few_shot: scored rooms (comma-separated)")
    var_args.add_argument("--support_rooms", default="classroom", type=str,
                          help="few_shot: support rooms (comma-separated)")
    var_args.add_argument("--teacher_epochs", default=70, type=int)
    var_args.add_argument("--student_epochs", default=200, type=int)
    var_args.add_argument("--few_shot_ratio", default=0.1, type=float)
    var_args.add_argument("--kd_weight", default=0.3, type=float)
    #
    return var_args.parse_args()


def sections(var_text, var_header_re):
    """
    [description]
    : split a printed report into (header, block) pairs at the lines matching var_header_re, so the
      per-room metrics can be read without a second run.
    """
    #
    var_out, var_name, var_buf = [], None, []
    for var_line in var_text.splitlines():
        var_match = var_header_re.match(var_line.rstrip())
        if var_match is not None:
            if var_name is not None:
                var_out.append((var_name, "\n".join(var_buf)))
            var_name, var_buf = var_match.group(1), []
        elif var_name is not None:
            var_buf.append(var_line)
    if var_name is not None:
        var_out.append((var_name, "\n".join(var_buf)))
    return [var_item for var_item in var_out if var_item[0] not in SKIP_HEADERS]


def run_process(var_cmd):
    """
    [description]
    : run one cell and return (stdout+stderr, returncode). Prints the tail on a non-zero exit so a
      broken cell is diagnosable from the sweep log.
    """
    #
    var_proc = subprocess.run(var_cmd, capture_output=True, text=True)
    var_out = var_proc.stdout + var_proc.stderr
    if var_proc.returncode != 0:
        print("  exit %d, last output:" % var_proc.returncode)
        print("  " + "\n  ".join(var_out.splitlines()[-15:]))
    return var_out, var_proc.returncode


def run_diag_cell(var_room, var_lr, var_bs, var_backbone, var_steps):
    """
    [description]
    : diag protocol: one room, the same-room diagnostic.
    """
    #
    var_cmd = [sys.executable, DIAG, "--room", var_room, "--backbone", var_backbone,
               "--steps", str(var_steps), "--lr", str(var_lr), "--batch_size", str(var_bs)]
    var_out, var_rc = run_process(var_cmd)
    var_best = BEST_RE.search(var_out)
    if var_best is None:
        return None
    var_centroid = CENTROID_RE.search(var_out)
    var_constant = CONSTANT_RE.search(var_out)
    var_acc_centroid = ACC_CENTROID_RE.search(var_out)
    var_acc_model = ACC_MODEL_RE.search(var_out)
    return {
        "room": var_room, "lr": str(var_lr), "batch": int(var_bs),
        "constant": float(var_constant.group(1)) if var_constant else None,
        "centroid": float(var_centroid.group(1)) if var_centroid else None,
        "best": float(var_best.group(1)), "best_step": int(var_best.group(2)),
        "last": float(var_best.group(3)),
        "ema": float(var_best.group(4)) if var_best.group(4) else None,
        "acc_centroid": float(var_acc_centroid.group(1)) if var_acc_centroid else None,
        "acc_model": float(var_acc_model.group(1)) if var_acc_model else None,
    }


def run_cross_domain_cell(var_lr, var_bs, var_args, var_hp):
    """
    [description]
    : cross_domain protocol: train the ResNet1D localizer on --train_envs, score every other room, and
      read the per-room MDE and constant baseline off the printed report.
    """
    #
    var_cmd = [sys.executable, CROSS_DOMAIN, "--model", "resnet1d", "--task", "location",
               "--train_envs", var_args.train_envs, "--repeat", str(var_args.repeat),
               "--epochs", str(var_args.epochs),
               "--hp", f"hwild.lr={var_lr}", "--hp", f"hwild.batch_size={var_bs}"]
    for var_item in var_hp:
        var_cmd += ["--hp", var_item]
    var_out, var_rc = run_process(var_cmd)
    var_per_room = {}
    for var_name, var_block in sections(var_out, CD_HEADER_RE):
        var_mde = CD_MDE_RE.search(var_block)
        if var_mde is None:
            continue
        var_const = CD_CONST_RE.search(var_block)
        var_per_room[var_name] = {"error": float(var_mde.group(1)),
                                  "const": float(var_const.group(1)) if var_const else None}
    if not var_per_room:
        return None
    return _summarize(var_lr, var_bs, var_per_room, "error", "const")


def run_few_shot_cell(var_lr, var_bs, var_args, var_hp):
    """
    [description]
    : few_shot protocol: teacher on --train_rooms, student on --support_rooms, scored on --test_rooms;
      read the student/teacher MDE and the training-room constant off the printed report.
    """
    #
    var_cmd = [sys.executable, FEW_SHOT, "--backbone", var_args.backbone,
               "--train_rooms", var_args.train_rooms, "--test_rooms", var_args.test_rooms,
               "--support_rooms", var_args.support_rooms,
               "--repeat", str(var_args.repeat), "--standardize", "window",
               "--teacher_epochs", str(var_args.teacher_epochs),
               "--student_epochs", str(var_args.student_epochs),
               "--few_shot_ratio", str(var_args.few_shot_ratio),
               "--kd_weight", str(var_args.kd_weight),
               "--hp", f"hwild.lr={var_lr}", "--hp", f"hwild.batch_size={var_bs}"]
    for var_item in var_hp:
        var_cmd += ["--hp", var_item]
    var_out, var_rc = run_process(var_cmd)
    var_per_room = {}
    for var_name, var_block in sections(var_out, FS_HEADER_RE):
        var_student = FS_STUDENT_RE.search(var_block)
        if var_student is None:
            continue
        var_teacher = FS_TEACHER_RE.search(var_block)
        var_const = FS_CONST_RE.search(var_block)
        var_per_room[var_name.lower()] = {
            "error": float(var_student.group(1)),
            "const": float(var_const.group(1)) if var_const else None,
            "teacher": float(var_teacher.group(1)) if var_teacher else None}
    if not var_per_room:
        return None
    return _summarize(var_lr, var_bs, var_per_room, "error", "const", "teacher")


def _summarize(var_lr, var_bs, var_per_room, var_key, var_const_key, var_teacher_key=None):
    """
    [description]
    : the mean metric over the scored rooms plus a per-room record, for a real-protocol cell.
    """
    #
    var_errors = [var_record[var_key] for var_record in var_per_room.values()]
    var_consts = [var_record[var_const_key] for var_record in var_per_room.values()
                  if var_record.get(var_const_key) is not None]
    var_teachers = [var_record[var_teacher_key] for var_record in var_per_room.values()
                    if var_record.get(var_teacher_key) is not None] if var_teacher_key else []
    return {
        "lr": str(var_lr), "batch": int(var_bs), "per_room": var_per_room,
        "mean_error": sum(var_errors) / len(var_errors),
        "mean_const": (sum(var_consts) / len(var_consts)) if var_consts else None,
        "mean_teacher": (sum(var_teachers) / len(var_teachers)) if var_teachers else None,
    }


def format_diag_table(var_rows):
    """
    [description]
    : aligned table for the diag protocol rows.
    """
    #
    var_lines = [f"{'lr':>7} {'batch':>6} {'constant':>9} {'centroid':>9} {'best':>8} "
                 f"{'@step':>7} {'last':>8} {'ema':>8} {'accCent':>8} {'accModel':>9}"]
    for var_r in var_rows:
        var_lines.append(f"{var_r['lr']:>7} {var_r['batch']:>6} "
                         f"{_fmt(var_r['constant'], '.3f'):>9} {_fmt(var_r['centroid'], '.3f'):>9} "
                         f"{var_r['best']:>8.3f} {var_r['best_step']:>7d} {var_r['last']:>8.3f} "
                         f"{_fmt(var_r['ema'], '.3f'):>8} {_fmt(var_r['acc_centroid'], '.2f'):>8} "
                         f"{_fmt(var_r['acc_model'], '.2f'):>9}")
    return "\n".join(var_lines)


def format_real_table(var_rows, var_has_teacher):
    """
    [description]
    : aligned table for a real-protocol cell: the mean metric, the mean constant baseline, and each
      room's MDE.
    """
    #
    var_rooms = sorted({var_room for var_r in var_rows for var_room in var_r["per_room"]})
    var_title = "student" if var_has_teacher else "MDE"
    var_head = f"{'lr':>7} {'batch':>6} {var_title:>9} {'const':>9}"
    if var_has_teacher:
        var_head += f" {'teacher':>9}"
    var_head += "  " + " ".join(f"{var_room[:10]:>10}" for var_room in var_rooms)
    var_lines = [var_head]
    for var_r in var_rows:
        var_line = (f"{var_r['lr']:>7} {var_r['batch']:>6} {var_r['mean_error']:>9.3f} "
                    f"{_fmt(var_r['mean_const'], '.3f'):>9}")
        if var_has_teacher:
            var_line += f" {_fmt(var_r['mean_teacher'], '.3f'):>9}"
        var_line += "  " + " ".join(
            f"{var_r['per_room'][var_room]['error']:>10.3f}" if var_room in var_r["per_room"]
            else f"{'-':>10}" for var_room in var_rooms)
        var_lines.append(var_line)
    return "\n".join(var_lines)


def _fmt(var_value, var_spec):
    """
    [description]
    : format a possibly-None float as a fixed-width string.
    """
    #
    return "-" if var_value is None else format(var_value, var_spec)


def room_diagnosis(var_rows, var_baseline_lr, var_baseline_batch):
    """
    [description]
    : the diag-protocol verdict for one room.
    """
    #
    var_best_row = min(var_rows, key=lambda var_r: var_r["best"])
    var_base_row = next((var_r for var_r in var_rows
                         if var_r["lr"] == str(var_baseline_lr)
                         and var_r["batch"] == var_baseline_batch), None)
    var_lines = [f"  best config: lr={var_best_row['lr']} batch={var_best_row['batch']} -> "
                 f"best valid {var_best_row['best']:.3f} m @ step {var_best_row['best_step']}"]
    if var_base_row is not None:
        var_gain = var_base_row["best"] - var_best_row["best"]
        var_lines.append(f"  current lr={var_baseline_lr} batch={var_baseline_batch} -> "
                         f"best valid {var_base_row['best']:.3f} m "
                         f"(sweep {'improves' if var_gain > 0 else 'does not improve'}: "
                         f"{var_gain:+.3f} m)")
    if var_best_row["centroid"] is not None:
        if var_best_row["best"] <= var_best_row["centroid"] + 0.02:
            var_lines.append(f"  reaches the nearest-centroid ceiling "
                             f"({var_best_row['centroid']:.3f} m): yes")
        else:
            var_lines.append(f"  reaches the nearest-centroid ceiling "
                             f"({var_best_row['centroid']:.3f} m): NO "
                             f"(+{var_best_row['best'] - var_best_row['centroid']:.3f} m) -- the data "
                             f"may be the limit here, not the optimiser")
    if var_best_row["ema"] is not None:
        var_verdict = ("EMA helps" if var_best_row["ema"] < var_best_row["best"]
                       else "keep the checkpoint")
        var_lines.append(f"  weight-EMA {var_best_row['ema']:.3f} m vs best checkpoint "
                         f"{var_best_row['best']:.3f} m -> {var_verdict}")
    return "\n".join(var_lines)


def real_diagnosis(var_rows, var_baseline_lr, var_baseline_batch, var_has_teacher):
    """
    [description]
    : the real-protocol verdict: the winning config, its gain over the current one, and whether the
      mean metric stays under the mean constant baseline.
    """
    #
    var_best_row = min(var_rows, key=lambda var_r: var_r["mean_error"])
    var_base_row = next((var_r for var_r in var_rows
                         if var_r["lr"] == str(var_baseline_lr)
                         and var_r["batch"] == var_baseline_batch), None)
    var_lines = [f"  best config: lr={var_best_row['lr']} batch={var_best_row['batch']} -> "
                 f"mean MDE {var_best_row['mean_error']:.3f} m"]
    if var_base_row is not None:
        var_gain = var_base_row["mean_error"] - var_best_row["mean_error"]
        var_lines.append(f"  current lr={var_baseline_lr} batch={var_baseline_batch} -> "
                         f"mean MDE {var_base_row['mean_error']:.3f} m "
                         f"(sweep {'improves' if var_gain > 0 else 'does not improve'}: "
                         f"{var_gain:+.3f} m)")
    if var_best_row["mean_const"] is not None:
        var_verdict = ("beats" if var_best_row["mean_error"] < var_best_row["mean_const"] else "does "
                       "NOT beat")
        var_lines.append(f"  {var_verdict} the mean constant baseline "
                         f"({var_best_row['mean_const']:.3f} m)")
    if var_has_teacher and var_best_row["mean_teacher"] is not None:
        var_lines.append(f"  teacher mean MDE {var_best_row['mean_teacher']:.3f} m "
                         f"(the same-protocol non-few-shot reference)")
    return "\n".join(var_lines)


def main():
    """
    [description]
    : run the sweep and write the diagnosis file.
    """
    #
    var_args = parse_args()
    var_lrs = var_args.lrs.split()
    var_batches = [int(var_b) for var_b in var_args.batches.split()]
    var_hp = list(var_args.hp or [])
    if not var_args.no_synth and var_args.protocol in ("cross_domain", "few_shot"):
        var_hp.append("data.synth_wimans.enabled=true")
    var_out = var_args.out or os.path.join(
        "output", "diagnose", f"{var_args.protocol}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    os.makedirs(os.path.dirname(var_out) or ".", exist_ok=True)
    #
    var_header = ["=" * 78,
                  "LOCALIZER lr x batch DIAGNOSIS - protocol %s" % var_args.protocol,
                  "generated %s" % datetime.now().isoformat(timespec="seconds"),
                  "backbone %s | lr swept: %s | batch swept: %s"
                  % (var_args.backbone, ", ".join(var_lrs),
                     ", ".join(str(var_b) for var_b in var_batches)),
                  "=" * 78]
    #
    var_all_rows = []
    if var_args.protocol == "diag":
        var_rows_by_room = {var_room: [] for var_room in var_args.rooms.split()}
        for var_room in var_rows_by_room:
            for var_lr in var_lrs:
                for var_bs in var_batches:
                    print("[%s] lr=%s batch=%s ..." % (var_room, var_lr, var_bs), flush=True)
                    var_row = run_diag_cell(var_room, var_lr, var_bs, var_args.backbone,
                                            var_args.steps)
                    if var_row is None:
                        print("[%s] lr=%s batch=%s FAILED" % (var_room, var_lr, var_bs))
                        continue
                    print("[%s] lr=%s batch=%s -> best valid %.3f m @ step %d"
                          % (var_room, var_lr, var_bs, var_row["best"], var_row["best_step"]))
                    var_rows_by_room[var_room].append(var_row)
                    var_all_rows.append(var_row)
        var_lines = var_header + [
            "", "How to read it: 'best' is the best-validation error the optimiser reached (the number",
            "to compare), 'last' is the final step (what you get without checkpoint selection), 'ema'",
            "is the weight-averaged model. Lower 'best' beats the room's nearest-centroid baseline ->",
            "the optimiser was the limit; still above it -> that room's representation/data is the limit.",
        ]
        for var_room in var_rows_by_room:
            var_rows = var_rows_by_room[var_room]
            var_lines += ["", "--- %s ---" % var_room]
            if not var_rows:
                var_lines.append("  no successful cell")
                continue
            var_lines.append(format_diag_table(var_rows))
            var_lines.append(room_diagnosis(var_rows, var_args.baseline_lr, var_args.baseline_batch))
        var_csv_keys = ("room", "lr", "batch", "constant", "centroid", "best", "best_step", "last",
                        "ema", "acc_centroid", "acc_model")
    else:
        var_is_few_shot = var_args.protocol == "few_shot"
        var_runner = run_few_shot_cell if var_is_few_shot else run_cross_domain_cell
        for var_lr in var_lrs:
            for var_bs in var_batches:
                print("[%s] lr=%s batch=%s ..." % (var_args.protocol, var_lr, var_bs), flush=True)
                var_row = var_runner(var_lr, var_bs, var_args, var_hp)
                if var_row is None:
                    print("[%s] lr=%s batch=%s FAILED" % (var_args.protocol, var_lr, var_bs))
                    continue
                print("[%s] lr=%s batch=%s -> mean MDE %.3f m (const %s, teacher %s)"
                      % (var_args.protocol, var_lr, var_bs, var_row["mean_error"],
                         _fmt(var_row["mean_const"], ".3f"), _fmt(var_row["mean_teacher"], ".3f")))
                var_all_rows.append(var_row)
        var_metric_label = ("mean student MDE" if var_is_few_shot else "mean MDE")
        var_lines = var_header + [
            "", "How to read it: %s is the metric the protocol reports, averaged over the scored"
            % var_metric_label,
            "rooms; 'const' is the same run's constant baseline and 'teacher' the few-shot run's",
            "same-protocol non-few-shot reference. A config is worse if it raises the mean, and a",
            "cross-room number below the constant is the only evidence of transfer.",
        ]
        if var_all_rows:
            var_lines += ["", format_real_table(var_all_rows, var_is_few_shot)]
            var_lines.append(real_diagnosis(var_all_rows, var_args.baseline_lr,
                                            var_args.baseline_batch, var_is_few_shot))
        else:
            var_lines.append("  no successful cell")
        var_csv_keys = ("lr", "batch", "room", "error", "const", "teacher")
    #
    var_text = "\n".join(var_lines) + "\n"
    with open(var_out, "w") as var_file:
        var_file.write(var_text)
    var_csv = os.path.splitext(var_out)[0] + ".csv"
    with open(var_csv, "w") as var_file:
        var_file.write(",".join(var_csv_keys) + "\n")
        if var_args.protocol == "diag":
            for var_row in var_all_rows:
                var_file.write(",".join(
                    "" if var_row.get(var_key) is None else str(var_row.get(var_key))
                    for var_key in var_csv_keys) + "\n")
        else:
            ## real protocols: one CSV row per scored room, so the per-room spread is not lost when
            ## only the mean is plotted
            for var_row in var_all_rows:
                for var_room, var_record in sorted(var_row["per_room"].items()):
                    var_file.write(",".join((
                        var_row["lr"], str(var_row["batch"]), var_room,
                        _fmt(var_record.get("error"), ".4f"),
                        _fmt(var_record.get("const"), ".4f"),
                        _fmt(var_record.get("teacher"), ".4f"))) + "\n")
    #
    print("\n" + var_text)
    print("Diagnosis written to %s" % var_out)
    print("CSV written to       %s" % var_csv)


if __name__ == "__main__":
    #
    var_start = time.time()
    main()
    print("Total time: %s seconds" % (time.time() - var_start))
