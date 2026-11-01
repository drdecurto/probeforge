S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs
for i in $(seq 1 2880); do
  st=$(squeue -h -j 47808930 -o %T 2>/dev/null)
  [ "$st" != PENDING ] && { echo "bsccase gsmml state '$st', no fallback needed $(date)"; exit 0; }
  grep -q "CHAIN DONE" $L/phi4mini_followup.log 2>/dev/null && break; sleep 60; done
scancel 47808930; echo "cancelled bsccase gsmml $(date)"
for i in $(seq 1 720); do J=$(sbatch --parsable $L/gsmml_debug.sbatch 2>/dev/null) && break; sleep 120; done; echo "submitted gsmml on debug $J $(date)"
