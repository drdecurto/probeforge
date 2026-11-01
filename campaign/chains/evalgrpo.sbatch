#!/usr/bin/env bash
#SBATCH --job-name=cf-evalgrpo
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; E=$S/evalgrpo; D=$S/data/contrast13m_200; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
for sh in 0 1 2 3; do [ -f $E/test_grpo8b.shard$sh.jsonl.manifest.json ] || srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/c$sh.out --error=$E/c$sh.err \
  contrastforge collect --inputs $D/test_eval12.jsonl --only-ids $D/eshard$sh.json --out $E/test_grpo8b.shard$sh.jsonl --model $S/models/merged/v4c-8b_s1 --adapter $S/runs/grpo_8b --budgets 512 --samples 1 < /dev/null & done; wait
