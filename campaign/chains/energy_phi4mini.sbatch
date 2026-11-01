#!/usr/bin/env bash
#SBATCH --job-name=energyp4m
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
B=/gpfs/scratch/bsc21/bsc021885/cf/energybench; NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
srun --nodes=1 --ntasks=1 -w ${NODES[0]} --exact --gres=gpu:4 --cpus-per-task=80 env SLOTS=$B/slots_phi4mini.txt OUTD=$B/out EVALPY=$B/eval_patched.py $B/node_run2.sh &
srun --nodes=1 --ntasks=1 -w ${NODES[1]} --exact --gres=gpu:4 --cpus-per-task=80 env SLOTS=$B/slots_phi4mini.txt OUTD=$B/out_chat EVALPY=$B/eval_chat.py $B/node_run2.sh &
wait; echo "end $(date)"
