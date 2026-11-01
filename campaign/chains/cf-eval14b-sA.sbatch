#!/usr/bin/env bash
#SBATCH --job-name=cf-eval14b-sA
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=$S/eval14b; cd ~/probeforge
gpu() { for n in $2; do CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_${n}_s$3.out --error=$E/collect_${n}_s$3.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$3.json --out $E/test_$n.shard$3.jsonl --model $S/models/Qwen3-14B --adapter $S/runs/sft14b_$n --budgets 512 --samples 1; done; }
gpu 0 "random_s1 mined_s1" 0 & gpu 1 "random_s1 mined_s1" 1 & gpu 2 "random_s1 mined_s1" 2 & gpu 3 "random_s1 mined_s1" 3 &
wait
for n in random_s1 mined_s1; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok"; done
echo "end $(date)"
