#!/usr/bin/env bash
#SBATCH --job-name lor_all
#SBATCH --cpus-per-task 8
#SBATCH --mem 40G
#SBATCH --qos normal
#SBATCH --time 12:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --output lor_all_%j.out

#
## Pooled multi-user room-agnostic leave-one-room-out over WiMANS + H-WILD.
##
## Stage the H-WILD dataset once on a login node (compute nodes usually have no network):
##     git clone --depth 1 \
##       https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset.git dataset/hwild
##     rm -rf dataset/hwild/.git
##
## Then:  sbatch scripts/lor_all.sh
##
## Runs the general set-prediction model (--model set). Switch to the density head with
## --model density. --max_per_room 0 uses every sample/window of every room; drop it (or set a
## number) to cap for a quicker run.
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
export WANDB_MODE=offline

mkdir -p ./output/lor_all

start=$(date +%s)
echo "Start: $(date)"

python scripts/run_lor_all.py \
    --model set \
    --folds all \
    --max_per_room 0 \
    --epochs 100 \
    --repeat 3 \
    > ./output/lor_all/lor_all_set_full.txt 2>&1

end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
