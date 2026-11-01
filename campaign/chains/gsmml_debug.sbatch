#!/usr/bin/env bash
#SBATCH --job-name=gsmml
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=/gpfs/scratch/bsc21/bsc021885/cf/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMEXPR_MAX_THREADS=64 VLLM_WORKER_MULTIPROC_METHOD=spawn
k=0; for g in "en,es,ca,ar" "lb,eu,fr" "pt,it,de" "ru,zh-cn,zh-tw"; do
  CUDA_VISIBLE_DEVICES=$k /gpfs/scratch/bsc21/bsc021885/cf/envs/lmeval/bin/python /gpfs/scratch/bsc21/bsc021885/cf/v4/gsm_ml.py --langs $g --out /gpfs/scratch/bsc21/bsc021885/cf/v4/gsm --model /gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-14B --n 3000 > /gpfs/scratch/bsc21/bsc021885/cf/v4/gsm/log_gpu$k.txt 2>&1 &
  k=$((k+1)); done; wait; grep -h "^[a-z-]* {" /gpfs/scratch/bsc21/bsc021885/cf/v4/gsm/log_gpu*.txt; wc -l /gpfs/scratch/bsc21/bsc021885/cf/v4/gsm/*.jsonl
