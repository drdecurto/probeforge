#!/usr/bin/env bash
#SBATCH --job-name=stdv4c3
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); echo "nodes ${NODES[*]} $(date)"; for n in ${NODES[*]}; do srun --nodes=1 --ntasks=1 -w $n --exact --gres=gpu:4 --cpus-per-task=80 /gpfs/scratch/bsc21/bsc021885/cf/stdbench/node_std.sh & done; wait; echo "end $(date)"
