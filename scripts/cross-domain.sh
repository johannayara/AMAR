#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:v100:2

# Fail fast on errors
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
model=resnet1
mkdir -p "./output/cd/sim_wimans/${model}/"
start=$(date +%s)

export WANDB_MODE=offline 
python scripts/run_cross_domain.py --model "${model}" --task location --train_envs "meeting_room_synth, empty_room_synth, classroom_synth"\
 --hp density.count_rule=layout_free --hp data.synth_wimans.enabled=true\
 --epochs 70 > "./output/cd/sim_wimans/${model}/run_$(date +%Y%m%d_%H%M).log" 2>&1 #
 
#python scripts/run_cross_domain.py --model density_map --task location --repeat 5 --env "$env" --epochs 50 > "./output/density_map/${env}/test_2_cd.txt" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"