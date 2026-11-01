#!/usr/bin/env bash
#SBATCH --job-name=grpo8b
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NUMEXPR_MAX_THREADS=64 TOKENIZERS_PARALLELISM=false
$S/envs/grpo/bin/python -m accelerate.commands.launch --num_processes 4 --mixed_precision bf16 $S/grpo/grpo_train.py --model $S/models/merged/v4c-8b_s1 --out $S/runs/grpo_8b --train $S/grpo/pool8b/grpo_train_8b.jsonl --n-prompts 100000 --max-steps 600 --save-steps 25
echo "end $(date)"
