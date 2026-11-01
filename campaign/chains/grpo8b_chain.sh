S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
for c in 1 2 3; do [ -f $S/runs/grpo_8b/metrics.json ] && break; J=$(submit $L/grpo8b.sbatch); echo "== grpo8b round$c $J $(date)"; waitjob $J; ls $S/runs/grpo_8b | grep checkpoint | tail -2; done
echo "CHAIN DONE $(date)"
