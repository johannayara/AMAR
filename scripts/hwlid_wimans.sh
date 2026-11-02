#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:v100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

mkdir -p "./output/few_shot_hwild_wimans/group"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
export AMAR_ENCODER_NORM=group
## python scripts/run_hwild_to_wimans.py --repeat 3 --epochs 200 --train_rooms "Conference, Laboratory, empty_room, classroom" --test_rooms "meeting_room, Lounge, Office" > "./output/hwild_wimans/run_$(date +%Y%m%d_%H%M).log" 2>&1

## Synth in the test set: --hp enables the second WiMANS-format source for this run, so the *_synth
## room keys resolve. --test_rooms adds the synth counterpart of each scored real room (real rooms
## first); --support_rooms stays on the real classroom, so no synth frame is ever fine-tuned on or
## selects the checkpoint -- the synth rooms are scored only. The localizer runner had no --hp until
## now, so this is the switch that turns the source on for a run.
# python scripts/run_few_shot_hwild_to_wimans.py \
#   --train_rooms "Conference, Laboratory, Lounge, Office" --test_rooms "meeting_room, empty_room" \
#   --support_rooms classroom --repeat 3 \
#   --teacher_epochs 200 --student_epochs 200 --few_shot_ratio 0.1 --kd_weight 0 > "./output/few_shot_hwild_wimans/group/run_$(date +%Y%m%d_%H%M).log" 2>&1
python scripts/run_few_shot_hwild_to_wimans.py \
  --train_rooms "Conference, Laboratory, Lounge, Office" \
  --test_rooms "meeting_room, empty_room, meeting_room_synth, empty_room_synth" \
  --support_rooms classroom --repeat 3 \
  --teacher_epochs 200 --student_epochs 200 --few_shot_ratio 0.1 --kd_weight 0 \
  --hp data.synth_wimans.enabled=true \
  > "./output/few_shot_hwild_wimans/group/run_synth_$(date +%Y%m%d_%H%M).log" 2>&1


# python scripts/run_few_shot_hwild_to_wimans.py --repeat 3 \
#   --teacher_epochs 200 --student_epochs 200 \
#   --train_rooms "Conference, Laboratory, Lounge, Office" \
#   --test_rooms "meeting_room, classroom, empty_room" \
#   --few_shot_ratio 0.1 --kd_weight 0.1 > "./output/few_shot_hwild_wimans/group/run_$(date +%Y%m%d_%H%M).log" 2>&1 

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"