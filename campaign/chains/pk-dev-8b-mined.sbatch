#!/usr/bin/env bash
#SBATCH --job-name=pk-dev-8b-mined
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; P=/gpfs/scratch/bsc21/bsc021885/cf/data/packs; E=/gpfs/scratch/bsc21/bsc021885/cf/evalpacks/dev112; mkdir -p $E; cd ~/probeforge
A=""; [ -n "/gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_mined_s1" ] && A="--adapter /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_mined_s1"
for K in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_mined8b_s$K.out --error=$E/collect_mined8b_s$K.err \
  contrastforge collect --inputs /gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_coverage_extension_2026-10-13/development112.jsonl --only-ids /gpfs/scratch/bsc21/bsc021885/cf/data/packs/dev112_shard$K.json --out $E/test_mined8b.shard$K.jsonl --model /gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-8B $A --budgets 512 --samples 1 & done; wait
cat $E/test_mined8b.shard*.jsonl > $E/test_mined8b.jsonl; cp $E/test_mined8b.shard0.jsonl.manifest.json $E/test_mined8b.jsonl.manifest.json
contrastforge evaluate --inputs /gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_coverage_extension_2026-10-13/development112.jsonl --rollouts $E/test_mined8b.jsonl --out $E/report_mined8b.json >/dev/null && echo "report_mined8b ok"
echo "end $(date)"
