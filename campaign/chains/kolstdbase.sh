for i in $(seq 1 720); do J=$(sbatch --parsable --exclude=as04r5b02 /gpfs/scratch/bsc21/bsc021885/cf/logs/koleval.sbatch 2>/dev/null) && break; sleep 60; done; echo "koleval std-base $J $(date)"
