#!/usr/bin/env bash
#SBATCH --job-name=cf-prepare-pilot
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -euo pipefail
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast60
cd ~/probeforge
echo "start $(date) on $(hostname)"
for k in 0 1 2 3; do echo "shard$k: $(wc -l < $S/data/student_base.shard$k.jsonl)"; done
for arm in translated random mined no_counterfactual; do
  echo "--- prepare $arm"
  contrastforge prepare --inputs $D/train.jsonl $D/validation.jsonl \
    --rollouts $S/data/student_base.shard0.jsonl $S/data/student_base.shard1.jsonl $S/data/student_base.shard2.jsonl $S/data/student_base.shard3.jsonl \
    --strategy $arm --tokenizer $S/models/Qwen3-8B --token-budget 2000000 --out $S/data/curricula60/$arm 2>&1 | tail -15
done
ls $S/data/curricula60/mined
echo "--- 10-step pilot train (mined)"
srun --ntasks=1 --cpu-bind=none contrastforge train --config configs/contrast_sft_mn5.json \
  --model $S/models/Qwen3-8B --train-data $S/data/curricula60/mined/sft_train.jsonl \
  --validation-data $S/data/curricula60/mined/sft_validation.jsonl \
  --output-dir $S/runs/pilot10_mined --max-steps 10 2>&1 | tail -30
ls -la $S/runs/pilot10_mined
echo "end $(date)"
