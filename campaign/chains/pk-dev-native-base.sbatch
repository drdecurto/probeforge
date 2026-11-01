#!/usr/bin/env bash
#SBATCH --job-name=pk-dev-native-base-base
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; P=/gpfs/scratch/bsc21/bsc021885/cf/data/packs; V=/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_coverage_extension_2026-10-13; E=$S/evalpacks/dev112native; mkdir -p $E; cd ~/probeforge
run() { CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$2_s$3.out --error=$E/collect_$2_s$3.err \
  contrastforge collect --inputs $V/development112_native.jsonl --only-ids $P/dev112_shard$3.json --out $E/test_$2.shard$3.jsonl --model $S/models/Qwen3-8B $4 --budgets 512 --samples 1; }
run 0 base8b 0 "" & run 1 base8b 1 "" & run 2 base8b 2 "" & run 3 base8b 3 "" &
wait
for n in base8b; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; done
echo "end $(date)"
