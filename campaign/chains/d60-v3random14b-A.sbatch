#!/usr/bin/env bash
#SBATCH --job-name=d60-v3random14b-A
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; P=/gpfs/scratch/bsc21/bsc021885/cf/data/packs; C=/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_eval_candidate_2026-10-13; E=/gpfs/scratch/bsc21/bsc021885/cf/evalpacks/diag60; cd ~/probeforge
A=""; [ -n "/gpfs/scratch/bsc21/bsc021885/cf/runs/sftv3_14b_random" ] && A="--adapter /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv3_14b_random"; K=0
for sh in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_v3random14b_s$sh.out --error=$E/collect_v3random14b_s$sh.err \
  contrastforge collect --inputs $C/test_diagnostic60.jsonl --only-ids $P/diag60_shard$sh.json --out $E/test_v3random14b.shard$sh.jsonl --model /gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-14B $A --budgets 512 --samples 1 & K=$((K+1)); done; wait
echo "end $(date)"
