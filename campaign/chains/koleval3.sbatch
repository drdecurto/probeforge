#!/usr/bin/env bash
#SBATCH --job-name=kol-eval3
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclude=as04r5b02
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); i=0
for spec in $KSPECS3; do srun --nodes=1 --ntasks=1 -w ${NODES[$i]} --exact --cpus-per-task=80 --gres=gpu:4 --cpu-bind=none bash -c "export KSPECS='$spec'; bash $S/kolibri/serve_eval.sh" < /dev/null & i=$((i+1)); done; wait; echo "end $(date)"
