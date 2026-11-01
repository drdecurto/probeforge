#!/usr/bin/env bash
#SBATCH --job-name=energychat
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
B=/gpfs/scratch/bsc21/bsc021885/cf/energybench; mapfile -t SLOTS < $B/slots.txt
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); echo "nodes ${NODES[*]} $(date)"
for i in 0 1 2; do srun --nodes=1 --ntasks=1 -w ${NODES[$i]} --exact --gres=gpu:4 --cpus-per-task=80 env SLOTS=$B/slots_chat.txt OUTD=$B/out_chat EVALPY=$B/eval_chat.py $B/node_run2.sh & done
wait; echo "end $(date)"
