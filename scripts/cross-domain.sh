#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 20G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:v100:2

# Fail fast on errors
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
model=density_map
mkdir -p "./output/cd/synth/${model}/"
start=$(date +%s)

export WANDB_MODE=offline 
python scripts/run_cross_domain.py --model "${model}" --task location --train_envs empty_room,classroom\
 --hp density.count_rule=layout_free --hp data.synth_wimans.enabled=true\
 --epochs 70 > "./output/cd/synth/${model}/run_$(date +%Y%m%d_%H%M).log" 2>&1 #
 
#python scripts/run_cross_domain.py --model density_map --task location --repeat 5 --env "$env" --epochs 50 > "./output/density_map/${env}/test_2_cd.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"