#!/usr/bin/env bash
#SBATCH --job-name=cf-evalphi4mini-A
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=/gpfs/scratch/bsc21/bsc021885/cf/evalphi4mini; M=/gpfs/scratch/bsc21/bsc021885/cf/models/Phi-4-mini-instruct; cd ~/probeforge
gpu() { for n in $2; do A=""; [ $n != base ] && A="--adapter $S/runs/sftphi4mini_$n"; [ -f $E/test_$n.shard$3.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_${n}_s$3.out --error=$E/collect_${n}_s$3.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$3.json --out $E/test_$n.shard$3.jsonl --model $M $A --budgets 512 --samples 1; done; }
gpu 0 "base random" 0 & gpu 1 "base random" 1 & gpu 2 "base random" 2 & gpu 3 "base random" 3 &
wait; echo "end $(date)"
