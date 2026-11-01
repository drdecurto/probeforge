#!/usr/bin/env bash
#SBATCH --job-name=sig
#SBATCH --account=bsc21
#SBATCH --qos=gp_debug
#SBATCH --partition=gpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
source ~/probeforge/slurm/environment.sh; cd ~/probeforge; python /gpfs/scratch/bsc21/bsc021885/cf/sig/sig.py
