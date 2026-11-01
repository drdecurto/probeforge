#!/usr/bin/env bash
#SBATCH --job-name=itdiag
#SBATCH --account=bsc21
#SBATCH --qos=acc_interactive
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=/gpfs/scratch/bsc21/bsc021885/cf/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NUMEXPR_MAX_THREADS=64 VLLM_WORKER_MULTIPROC_METHOD=spawn
/gpfs/scratch/bsc21/bsc021885/cf/envs/lmeval/bin/python /gpfs/scratch/bsc21/bsc021885/cf/v4/it_diag.py /gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-14B 2>/dev/null | grep '^{' > /gpfs/scratch/bsc21/bsc021885/cf/v4/it_diag.jsonl; wc -l /gpfs/scratch/bsc21/bsc021885/cf/v4/it_diag.jsonl
