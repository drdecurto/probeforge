#!/usr/bin/env bash
#SBATCH --job-name=cf-evalphi4s
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; E=$S/evalphi4s; T=$S/data/contrast13m_200/test_eval12.jsonl; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
while IFS='|' read name cfg seed; do [ -f $E/test_$name.jsonl.manifest.json ] || srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$name.out --error=$E/collect_$name.err \
  contrastforge collect --inputs $T --out $E/test_$name.jsonl --model $S/models/phi-4 --adapter $S/runs/sftphi4_$name --budgets 512 --samples 1 < /dev/null &
done < $S/v4/phi4_seed_runs.txt; wait
