#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --qos normal
#SBATCH --time 4:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-4

set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR
# parameters
model=density_map
env=classroom
## default = plain THAT; thatv = all three features; the three "no*" variants each drop one feature
backbones=(default thatv thatv-nostats thatv-notokens thatv-nopool)
backbone=${backbones[$SLURM_ARRAY_TASK_ID]}

## one directory per backbone, so the array tasks never write over each other
log_dir="./output/${model}/${backbone}/${env}"
mkdir -p "${log_dir}"

start=$(date +%s)
echo "Start: $(date) | backbone=${backbone}"
export WANDB_MODE=offline
python scripts/run_main.py --model "${model}" --task location --repeat 3 \
    --env "${env}" --hp nn.backbone="${backbone}" --epochs 50 \
  > "${log_dir}/run_$(date +%Y%m%d_%H%M%S).log" 2>&1
end=$(date +%s)
echo "Total runtime: $((end - start)) seconds"
