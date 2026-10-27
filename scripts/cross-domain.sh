#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 48G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:a100:2

# Fail fast on errors
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
env=meeting_room
mkdir -p "./output/cd/density_map_dem/${env}/"
start=$(date +%s)

export WANDB_MODE=offline 
python scripts/run_cross_domain.py --model density_map_dem --task location --train_envs empty_room,classroom\
 --hp density.count_rule=layout_free\
 --epochs 70 > "./output/cd/density_map_dem/${env}/run_$(date +%Y%m%d_%H%M)_sgd.log" 2>&1 #
#python scripts/run_cross_domain.py --model density_map --task location --repeat 5 --env "$env" --epochs 50 > "./output/density_map/${env}/test_2_cd.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"