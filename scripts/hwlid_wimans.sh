#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:v100:1

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR




mkdir -p "./output/hwild_wimans/"

start=$(date +%s)
echo "Start: $(date)"
export WANDB_MODE=offline 
python scripts/run_hwild_to_wimans.py --repeat 3 --epochs 200 --train_rooms "Conference, Laboratory, empty_room, classroom" --test_rooms "meeting_room, Lounge, Office" > "./output/hwild_wimans/run_$(date +%Y%m%d_%H%M).log" 2>&1

end=$(date +%s)

echo "Total runtime: $((end - start)) seconds"