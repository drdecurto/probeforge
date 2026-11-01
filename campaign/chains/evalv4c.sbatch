#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv4c
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; E=$S/evalv4c; T=$S/data/contrast13m_200/test_eval12.jsonl; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); i=0
while IFS='|' read name base cur cfg seed; do n=$((i/4))
  [ -f $E/test_$name.jsonl.manifest.json ] || srun --nodes=1 --ntasks=1 -w ${NODES[$n]} --exact --gres=gpu:1 --cpus-per-task=20 --cpu-bind=none --output=$E/collect_$name.out --error=$E/collect_$name.err \
     contrastforge collect --inputs $T --out $E/test_$name.jsonl --model $S/models/$base --adapter $S/runs/sftv4c_$name --budgets 512 --samples 1 &
  i=$((i+1)); done < $S/v4/v4c_runs.txt
wait; echo "end $(date)"
