#!/usr/bin/env bash
#SBATCH --cpus-per-task 3
#SBATCH --mem 30G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:v100:2

#
## Leave-one-room-out density-map cross-domain run. One array task per held-out room: train on the
## other two rooms and test on the held-out one. The occupancy threshold is calibrated on the
## training rooms' validation split only; the held-out room contributes no labels.
# all_envs=(empty_room meeting_room classroom)
# held_out=${all_envs[$SLURM_ARRAY_TASK_ID]}
# train_envs=""
# for env in "${all_envs[@]}"; do
#     if [ "$env" != "$held_out" ]; then
#         train_envs="${train_envs:+${train_envs},}${env}"
#     fi
# done

# mkdir -p "./output/cd/density_map/lor_${held_out}/"

# start=$(date +%s)
# WANDB_MODE=offline python scripts/run_cross_domain.py \
#     --model density_map --task location --repeat 5 --train_envs "$train_envs" --epochs 50 \
#     > "./output/cd/density_map/lor_${held_out}/test_cd.txt" 2>&1
# echo "Total runtime held-out $held_out: $(( $(date +%s) - start )) seconds"

#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

## NOTE: --train_envs expects a comma-separated *string*. A bash array passed as "$train_envs"
## expands to its first element only ("meeting_room"), which silently ran the one-room protocol
## instead of LOR. Keep this a string.
train_envs="meeting_room,classroom"
held_out="empty_room"

mkdir -p "./output/cd/density_map/lor_${held_out}/"

start=$(date +%s)
WANDB_MODE=offline python scripts/run_cross_domain.py \
    --model density_map --task location --repeat 5 --train_envs "$train_envs" --epochs 100 \
    > "./output/cd/density_map/lor_${held_out}/run_$(date +%Y%m%d_%H%M%S).log" 2>&1
echo "Total runtime held-out $held_out: $(( $(date +%s) - start )) seconds"

