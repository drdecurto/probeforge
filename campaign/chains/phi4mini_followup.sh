S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; B=$S/energybench; D=$S/stdbench
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
for i in $(seq 1 2880); do grep -q "CHAIN DONE" $L/phi4mini_chain.log 2>/dev/null && break; sleep 60; done
[ -f $S/runs/sftphi4mini_mined/metrics.json ] || { echo "no phi4mini adapters, stop"; exit 1; }
miss() { n=0; while IFS='|' read s b a; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do for o in out out_chat; do [ -s $B/$o/$s/$l.json ] || n=$((n+1)); done; done; done < $B/slots_phi4mini.txt; echo $n; }
for c in 1 2 3; do m=$(miss); echo "energy missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/energy_phi4mini.sbatch); echo "== energyp4m $J"; waitjob $J; done
grep -q "^phi-4-mini-mined|" $D/models.txt || printf 'phi-4-mini-mined|merged/phi-4-mini-mined|\nphi-4-mini-random|merged/phi-4-mini-random|\n' >> $D/models.txt
for c in 1 2; do n=0; for x in phi-4-mini-mined phi-4-mini-random; do for p in mc mgsm; do [ -f $D/out/$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break
  J=$(submit $L/stdp4m.sbatch); echo "== stdp4m $J"; waitjob $J; grep -E "merge| ok | FAIL" $L/stdp4m-$J.out | tail; done
python3 $D/aggregate.py | grep -E "model|phi-4-mini"
echo "CHAIN DONE $(date)"
