#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv2-seeds
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
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=/gpfs/scratch/bsc21/bsc021885/cf/evalv2; mkdir -p $E; cd ~/probeforge
echo "node $(hostname) start $(date)"
run() { CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$2.out --error=$E/collect_$2.err \
  contrastforge collect --inputs $T --out $E/test_$2.jsonl --model $S/models/Qwen3-8B --adapter $3 --budgets 512 --samples 1; }
run 0 random_s1 /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_random_s1 &
run 1 random_s2 /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_random_s2 &
run 2 mined_s1 /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_mined_s1 &
run 3 mined_s2 /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv2_mined_s2 &
wait
for n in random_s1 random_s2 mined_s1 mined_s2; do contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok"; done
echo "end $(date)"
