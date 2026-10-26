# Cross-domain leave-one-room-out (LOR) results — density_map

Model: `density_map`, task `location`, 5 repeats, 50 epochs.
Protocol: train on the two named rooms, evaluate on the single held-out room (no labels from the
held-out room are used anywhere).

Repository HEAD: `3f913642393e09f03f048dac1f25c9b5dce7b7ac` (branch `main`).

## Summary — one row per held-out room

The **Held-out** column is the room that was left out of training (and is the room the metrics are
computed on).

| Held-out (left out) | Trained on              | Exact-count Acc        | Count MAE          | Occupancy Acc      | Occupancy F1       | Per-count Acc (0/1/2/3/4/5)                    | Thr  | Source file                              |
|---------------------|-------------------------|------------------------|--------------------|--------------------|--------------------|------------------------------------------------|------|------------------------------------------|
| **empty_room**      | meeting_room, classroom | — not available —      | —                  | —                  | —                  | —                                              | —    | see note below                           |
| **meeting_room**    | empty_room, classroom   | 0.1163 ± 0.0045 (SE)   | 2.0278 ± 0.0314    | 0.5136 ± 0.0014    | 0.2232 ± 0.0102    | 0.000 / 0.368 / 0.000 / 0.000 / 0.000 / 0.000  | 0.63 | `lor_meeting_room/test_cd.txt`           |
| **classroom**       | empty_room, meeting_room| 0.1730 ± 0.0126 (SE)   | 1.8177 ± 0.0383    | 0.4731 ± 0.0026    | 0.3733 ± 0.0370    | 0.000 / 0.326 / 0.279 / 0.095 / 0.069 / 0.000  | 0.37 | `lor_classroom/test_cd.txt`              |

Reading the table: for each row the model never saw the held-out room; the metrics are on that room.
`Per-count Acc` is the recall per true count (0..5); `0` is the empty frame.

## Note on the missing empty_room LOR row

There is **no valid LOR run with `empty_room` held out**. The two files in
`output/cd/density_map/lor_empty_room/` are not LOR runs:

- `lor_empty_room/test_cd.txt` — header says
  `Training rooms: ['meeting_room'] (1693 train / 188 valid) | Test rooms: ['empty_room', 'classroom']`.
- `lor_empty_room/test_cd_1.txt` — same one-room cross-domain run (wandb summaries list
  `test_results_per_env/{empty_room,classroom}/...`).

Both are the **one-room cross-domain** protocol (train on `meeting_room` only, test on
`empty_room` + `classroom`), i.e. the same as `output/cd/density_map_meeting_room_cd.txt`. They are
therefore not usable as the "empty_room held out" LOR result.

For reference only (NOT LOR), that mislabeled run reports on the `empty_room` test room:
`Accuracy 0.2347 ± 0.0052 | Balanced 0.1238 ± 0.0027 | MAE 1.7974 ± 0.0110 | Occ Acc 0.5006 ± 0.0012 | Occ F1 0.2569 ± 0.0039`,
`Per-count 0:0.000, 1:0.743, 2:0.000, 3:0.000, 4:0.000, 5:0.000`, `Thr 0.69`.

## Other notes

- The two valid LOR runs (`lor_meeting_room`, `lor_classroom`) predate the balanced-accuracy metric:
  their `PER_ENV_RESULTS` blocks have no `Balanced-count Accuracy` line and use the old
  single-threshold decision. The (mislabeled) `lor_empty_room` file is from newer code and does have
  a `Balanced-count Accuracy` line — another sign it is a different protocol.
- Held-out performance is low across the board (exact-count accuracy 0.12–0.17, occupancy F1 0.22–0.37),
  consistent with the cross-domain gap; `Per-count 0` is 0.000 for both available rooms.
- To complete the LOR set you need to rerun with `empty_room` held out, i.e. train on
  `meeting_room,classroom` and test on `empty_room`
  (`scripts/run_cross_domain.py --model density_map --task location --train_envs meeting_room,classroom`).
