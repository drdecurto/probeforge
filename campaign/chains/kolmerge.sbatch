#!/usr/bin/env bash
#SBATCH --job-name=cf-kolmerge
#SBATCH --account=bsc21
#SBATCH --qos=gp_debug
#SBATCH --partition=gpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; K=$S/kolibri; source $K/env.sh
# PAIRS="adapter_dir|out_dir ..."
for p in $PAIRS; do IFS='|' read a o <<< "$p"; [ -f $o/.merged ] && continue; echo "merge $a -> $o $(date +%T)"; python $K/merge_lora.py $a $o && touch $o/.merged; done; echo "end $(date)"
