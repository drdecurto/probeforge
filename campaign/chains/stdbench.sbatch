#!/usr/bin/env bash
#SBATCH --job-name=stdbench
#SBATCH --account=bsc21
#SBATCH --qos=acc_bsccase
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
echo "node $(hostname) $(date)"; /gpfs/scratch/bsc21/bsc021885/cf/stdbench/node_std.sh; echo "end $(date)"
