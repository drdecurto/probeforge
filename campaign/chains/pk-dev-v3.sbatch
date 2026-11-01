#!/usr/bin/env bash
#SBATCH --job-name=pk-dev-v3
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; P=/gpfs/scratch/bsc21/bsc021885/cf/data/packs; V=/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_coverage_extension_2026-10-13; E=$S/evalpacks/dev112; cd ~/probeforge
run() { CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$2_s$3.out --error=$E/collect_$2_s$3.err \
  contrastforge collect --inputs $V/development112.jsonl --only-ids $P/dev112_shard$3.json --out $E/test_$2.shard$3.jsonl --model $S/models/Qwen3-8B --adapter $S/runs/sftv3_$4 --budgets 512 --samples 1; }
(run 0 v3random 0 random; run 0 v3random 1 random) & (run 1 v3random 2 random; run 1 v3random 3 random) & (run 2 v3mined 0 mined; run 2 v3mined 1 mined) & (run 3 v3mined 2 mined; run 3 v3mined 3 mined) &
wait
for n in v3random v3mined; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; done
echo "end $(date)"
