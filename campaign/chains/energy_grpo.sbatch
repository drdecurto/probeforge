#!/usr/bin/env bash
#SBATCH --job-name=energygrpo
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
for i in 0; do srun --nodes=1 --ntasks=1 -w ${NODES[$i]} --exact --gres=gpu:4 --cpus-per-task=80 env SLOTS=$B/slots_grpo.txt OUTD=$B/out EVALPY=$B/eval_patched.py $B/node_run2.sh & done
for i in 1; do srun --nodes=1 --ntasks=1 -w ${NODES[$i]} --exact --gres=gpu:4 --cpus-per-task=80 env SLOTS=$B/slots_grpo.txt OUTD=$B/out_chat EVALPY=$B/eval_chat.py $B/node_run2.sh & done
wait; echo "end $(date)"
