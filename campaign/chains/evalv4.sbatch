#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv4
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
gpu() { n=$1; m=$2; sh=$3; [ -f /gpfs/scratch/bsc21/bsc021885/cf/evalv4/test_$n.shard$sh.jsonl.manifest.json ] && return
  CUDA_VISIBLE_DEVICES=$4 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=/gpfs/scratch/bsc21/bsc021885/cf/evalv4/collect_${n}_s$sh.out --error=/gpfs/scratch/bsc21/bsc021885/cf/evalv4/collect_${n}_s$sh.err \
  contrastforge collect --inputs /gpfs/scratch/bsc21/bsc021885/cf/data/contrast13m_200/test_eval12.jsonl --only-ids /gpfs/scratch/bsc21/bsc021885/cf/data/contrast13m_200/eshard$sh.json --out /gpfs/scratch/bsc21/bsc021885/cf/evalv4/test_$n.shard$sh.jsonl --model $m --adapter /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv4_$n --budgets 512 --samples 1; }
for sh in 0 1 2 3; do ( gpu 8b /gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-8B $sh $sh; gpu phi4mini /gpfs/scratch/bsc21/bsc021885/cf/models/Phi-4-mini-instruct $sh $sh ) & done; wait
