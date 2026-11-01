S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; B=$S/energybench
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
missing() { n=0; while IFS='|' read slot base adapter; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $B/out/$slot/$l.json ] || n=$((n+1)); done; done < $B/slots.txt; echo $n; }
echo "START $(date)"
for i in $(seq 1 720); do grep -q "CHAIN DONE" $L/muse_chain3.log 2>/dev/null && break; sleep 60; done
echo "muse chain done, starting energy benchmark $(date)"
for c in 1 2 3 4 5; do m=$(missing); echo "missing $m slot-languages $(date)"; [ "$m" = 0 ] && break
  J=$(submit $L/energybench.sbatch); echo "== energybench round$c $J $(date)"; waitjob $J; grep -E "up on GPU|done|failed|complete" $L/energybench-$J.out | tail -40; done
echo "CHAIN DONE $(date)"
