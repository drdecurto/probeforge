#!/usr/bin/env bash
#SBATCH --job-name=probe2n-acc_interactive
#SBATCH --account=bsc21
#SBATCH --qos=acc_interactive
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
srun hostname; nvidia-smi -L | head -2; date
