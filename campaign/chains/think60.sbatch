#!/usr/bin/env bash
#SBATCH --job-name=cf-think60
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; C=/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_eval_candidate_2026-10-13; T=$C/test_diagnostic60.jsonl; E=/gpfs/scratch/bsc21/bsc021885/cf/evalthink60; cd ~/probeforge
run() { [ -f $E/test_$2.jsonl.manifest.json ] && return; CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$2.out --error=$E/collect_$2.err \
  contrastforge collect --inputs $T --only-ids $C/think60_ids.json --out $E/test_$2.jsonl --model $S/models/$3 $4 --budgets 4096 --samples 1 --thinking; }
run 0 mined8b Qwen3-8B "--adapter $S/runs/sftv2_mined_s1" & run 1 base14b Qwen3-14B "" & run 2 random14b Qwen3-14B "--adapter $S/runs/sft14b_random" & run 3 v3mined14b Qwen3-14B "--adapter $S/runs/sftv3_14b_mined" &
wait; echo "end $(date)"
