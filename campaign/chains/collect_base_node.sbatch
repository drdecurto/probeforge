#!/usr/bin/env bash
#SBATCH --job-name=cf-collect-base4
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast60
cd ~/probeforge
echo "node $(hostname) start $(date) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for K in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none \
    --output=$S/logs/collect-base4-shard$K.out --error=$S/logs/collect-base4-shard$K.err \
    contrastforge collect --inputs $D/train.jsonl $D/validation.jsonl --only-ids $D/shard$K.json \
      --out $S/data/student_base.shard$K.jsonl --model $S/models/Qwen3-8B --budgets 512 --samples 1 &
done
wait
echo "all shards finished $(date)"
