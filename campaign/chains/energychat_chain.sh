S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; B=$S/energybench
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
missing() { n=0; while IFS='|' read slot base adapter; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $B/out_chat/$slot/$l.json ] || n=$((n+1)); done; done < $B/slots_chat.txt; echo $n; }
echo "START $(date)"
for i in $(seq 1 1440); do grep -q "CHAIN DONE" $L/energybench_chain.log 2>/dev/null && break; sleep 60; done
for c in 1 2 3 4; do m=$(missing); echo "missing $m $(date)"; [ "$m" = 0 ] && break
  J=$(submit $L/energychat.sbatch); echo "== energychat round$c $J $(date)"; waitjob $J; grep -E "up on|done|failed|idle" $L/energychat-$J.out | tail -30; done
echo "CHAIN DONE $(date)"
