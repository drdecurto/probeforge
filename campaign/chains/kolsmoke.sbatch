#!/usr/bin/env bash
#SBATCH --job-name=cf-kolsmoke
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; K=$S/kolibri; source $K/env.sh; cd ~/probeforge; rm -rf $S/runs/kolsmoke_v4c
nvidia-smi -L; date
python -m contrastforge.cli train --config configs/vkol_smoke_v4c.json > $K/logs/smoke_v4c.out 2> $K/logs/smoke_v4c.err
echo "rc $? $(date)"; tail -5 $K/logs/smoke_v4c.out; tr '\r' '\n' < $K/logs/smoke_v4c.err | grep -E "Error|error|it\]|s/it|trainable|Traceback" | tail -12
nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null; grep -o '"trainable_parameters": [0-9]*' $S/runs/kolsmoke_v4c/run_manifest.json 2>/dev/null
