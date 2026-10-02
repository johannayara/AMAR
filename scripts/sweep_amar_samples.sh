#!/usr/bin/env bash
#SBATCH --cpus-per-task 1
#SBATCH --mem 24G
#SBATCH --qos normal
#SBATCH --time 6:00:00
#SBATCH --gres gpu:a100:1
#SBATCH --array=0-8

#
## Training-set-size study for AMAR.
##
## One array task per training-set size. The test split is fixed in run_main.master_splitter, so all
## tasks are evaluated on exactly the same test samples and the only thing that changes is how many
## samples the model trains on. Set ENV/REPEAT/TASK/MODEL to override the defaults, e.g.
##     ENV=meeting_room REPEAT=5 sbatch scripts/sweep_amar_samples.sh
##     MODEL=AMAR TASK=activity sbatch scripts/sweep_amar_samples.sh
##
## Defaults target AMAR_WO_RVQ on the location task. NOTE: run_AMAR (the RVQ model) is activity-only
## today, so keep TASK=activity when MODEL=AMAR. EPOCHS defaults to 300: your existing 300-epoch
## runs select their best checkpoint at epochs 209-299, so 20 would be badly undertrained.
#
set -euo pipefail

source /software/anaconda3/etc/profile.d/conda.sh
conda activate AMAR

## Training-set sizes to sweep (0 = all available, ~3010 for one room)
samples_list=(25 50 100 200 400 800 1600 3010 0)
samples=${samples_list[$SLURM_ARRAY_TASK_ID]}

model=${MODEL:-AMAR_WO_RVQ}
task=${TASK:-location}
env=${ENV:-classroom}
repeat=${REPEAT:-5}
epochs=${EPOCHS:-300}

sample_tag=$([ "$samples" -eq 0 ] && echo "all" || echo "$samples")
out_dir="./output/sweep/${model}/${task}/${env}"
mkdir -p "$out_dir"

start=$(date +%s)
echo "Start: $(date) | model=$model task=$task env=$env samples=$samples repeat=$repeat epochs=$epochs"
export WANDB_MODE=offline
python scripts/run_main.py \
    --model "$model" --task "$task" --repeat "$repeat" --env "$env" \
    --train_samples "$samples" --epochs "$epochs" \
    > "${out_dir}/n${sample_tag}.txt" 2>&1
echo "Done samples=$sample_tag: $(( $(date +%s) - start )) seconds"
