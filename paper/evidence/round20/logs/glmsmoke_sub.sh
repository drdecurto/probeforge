L=/gpfs/scratch/bsc21/bsc021885/cf/logs
for i in $(seq 1 720); do J=$(sbatch --parsable $L/glmsmoke.sbatch 2>/dev/null) && { echo "submitted $J $(date)"; break; }; sleep 60; done
