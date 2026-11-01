S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; D=$S/stdbench
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; sleep 120; done; echo "$J"; }
missing() { n=0; while IFS='|' read name b a; do for p in mc mgsm; do [ -f $D/out/$name/$p/DONE ] || n=$((n+1)); done; done < $D/models.txt; echo $n; }
echo "START $(date)"
:
squeue -h -j 47723616 -o %T | grep -q PENDING && scancel 47723616 && echo "cancelled bsccase stdbench 47723616"
for c in 1 2 3; do m=$(missing); echo "missing $m $(date)"; [ "$m" = 0 ] && break
  J=$(submit $L/stdbench_debug.sbatch); echo "== stdbench round$c $J $(date)"; waitjob $J; grep -E " ok | FAIL" $L/stdbench-$J.out | tail -40; done
echo "CHAIN DONE $(date)"
