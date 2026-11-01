#!/usr/bin/env bash
#SBATCH --job-name=grpo-smoke
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false
rm -rf $S/runs/grpo_smoke
$S/envs/grpo/bin/python -m accelerate.commands.launch --num_processes 4 --mixed_precision bf16 $S/grpo/grpo_train.py --model $S/models/merged/v4c-8b_s1 --out $S/runs/grpo_smoke --max-steps 6 --n-prompts 200 --save-steps 100
ls $S/runs/grpo_smoke; cat $S/runs/grpo_smoke/metrics.json
