#!/usr/bin/env bash
#SBATCH --job-name=cf-eval14b-mixed
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=/gpfs/scratch/bsc21/bsc021885/cf/data/contrast13m_200; T=$D/test_eval12.jsonl; E=/gpfs/scratch/bsc21/bsc021885/cf/eval14b; ET=/gpfs/scratch/bsc21/bsc021885/cf/evalthink; ET6=/gpfs/scratch/bsc21/bsc021885/cf/evalthink60; C=/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_eval_candidate_2026-10-13; M=/gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-14B; cd ~/probeforge
nt() { for n in mixed50 think100; do [ -f $E/test_$n.shard$2.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_${n}_s$2.out --error=$E/collect_${n}_s$2.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$2.json --out $E/test_$n.shard$2.jsonl --model $M --adapter $S/runs/sft14b_$n --budgets 512 --samples 1; done; }
th() { n=$2; [ -f $ET/test_$n.jsonl.manifest.json ] || CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$ET/collect_$n.out --error=$ET/collect_$n.err \
  contrastforge collect --inputs $T --only-ids $D/think_ids.json --out $ET/test_$n.jsonl --model $M --adapter $S/runs/sft14b_$n --budgets 4096 --samples 1 --thinking
  [ -f $ET6/test_$n.jsonl.manifest.json ] || CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$ET6/collect_$n.out --error=$ET6/collect_$n.err \
  contrastforge collect --inputs $C/test_diagnostic60.jsonl --only-ids $C/think60_ids.json --out $ET6/test_$n.jsonl --model $M --adapter $S/runs/sft14b_$n --budgets 4096 --samples 1 --thinking; }
(nt 0 0; th 0 mixed50) & (nt 1 1; th 1 think100) & nt 2 2 & nt 3 3 &
wait; echo "end $(date)"
