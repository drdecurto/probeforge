#!/usr/bin/env bash
#SBATCH --job-name=c60-mined8b-B
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; P=/gpfs/scratch/bsc21/bsc021885/cf/data/packs; C=/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_eval_candidate_2026-10-13; E=$S/evalpacks/core60; mkdir -p $E; cd ~/probeforge
A=""; [ -n "/gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_mined_s1" ] && A="--adapter /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_mined_s1"; K=0
for sh in 4 5 6 7; do CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_mined8b_s$sh.out --error=$E/collect_mined8b_s$sh.err \
  contrastforge collect --inputs $C/test_core60.jsonl --only-ids $P/core60_shard$sh.json --out $E/test_mined8b.shard$sh.jsonl --model /gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-8B $A --budgets 512 --samples 1 & K=$((K+1)); done; wait
echo "end $(date)"
