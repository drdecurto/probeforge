#!/usr/bin/env bash
#SBATCH --job-name=kol-eval
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
export KSPECS="kolbase|/gpfs/scratch/bsc21/bsc021885/cf/models/Kolibri-1-BF16"; /gpfs/scratch/bsc21/bsc021885/cf/kolibri/serve_eval.sh
