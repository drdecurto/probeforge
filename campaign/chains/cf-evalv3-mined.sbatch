#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv3-mined
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=$S/evalv2; cd ~/probeforge
for K in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_v3_mined_s$K.out --error=$E/collect_v3_mined_s$K.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$K.json --out $E/test_v3_mined.shard$K.jsonl --model $S/models/Qwen3-8B --adapter $S/runs/sftv3_mined --budgets 512 --samples 1 & done; wait
cat $E/test_v3_mined.shard*.jsonl > $E/test_v3_mined.jsonl; cp $E/test_v3_mined.shard0.jsonl.manifest.json $E/test_v3_mined.jsonl.manifest.json
contrastforge evaluate --inputs $T --rollouts $E/test_v3_mined.jsonl --out $E/report12_v3_mined.json >/dev/null && echo "report12_v3_mined ok"
echo "end $(date)"
