#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv3-14b-sA
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=/gpfs/scratch/bsc21/bsc021885/cf/data/contrast13m_200; T=$D/test_eval12.jsonl; E=/gpfs/scratch/bsc21/bsc021885/cf/eval14b; M=/gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-14B; cd ~/probeforge
gpu() { for n in $2; do [ -f $E/test_v3_$n.shard$3.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_v3${n}_s$3.out --error=$E/collect_v3${n}_s$3.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$3.json --out $E/test_v3_$n.shard$3.jsonl --model $M --adapter $S/runs/sftv3_14b_$n --budgets 512 --samples 1; done; }
gpu 0 "random_s2 mined_s2" 0 & gpu 1 "random_s2 mined_s2" 1 & gpu 2 "random_s2 mined_s2" 2 & gpu 3 "random_s2 mined_s2" 3 &
wait; echo "end $(date)"
