S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; K=$S/kolibri; D=$S/data/contrast13m_200
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable "$@" 2>/dev/null) && break; sleep 60; done; echo "$J"; }
echo "START $(date)"
for c in 1 2; do r=$(python $K/remaining.py kolbase $D/test_eval12.jsonl 1); rm -f $K/eval/kolbase_r*.json; echo "kolbase remaining $r"; [ "$r" = 0 ] && break; J=$(submit $L/koleval.sbatch); echo "== koleval $J $(date)"; waitjob $J; grep -E "remaining|done|failed" $L/kol-eval-$J.out | head; done
cat $K/eval/test_kolbase.r*.jsonl | python -c "
import sys,json
seen=set();out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id']);out.append(l)
sys.stdout.write(''.join(out))" > $K/eval/test_kolbase.jsonl; cp $(ls $K/eval/test_kolbase.r*.jsonl.manifest.json | head -1) $K/eval/test_kolbase.jsonl.manifest.json
contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $K/eval/test_kolbase.jsonl --out $K/eval/report12_kolbase.json < /dev/null > /dev/null
python -c "
import json;b=json.load(open('$K/eval/report12_kolbase.json'))['budgets']['512']
print('KOLBASE pass@1=%.3f worst=%.3f tok=%.0f len_stops=%s'%(b['pass_at_1'],b['worst_language'],b['mean_completion_tokens'],b.get('length_stops')))
print(' view',{k:round(v,2) for k,v in b['per_view'].items()}); print(' lang',{k:round(v,2) for k,v in sorted(b['per_language'].items())})"
echo "KOLPROBE DONE $(date)"
