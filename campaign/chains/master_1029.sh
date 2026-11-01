S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; Q=$S/q32; D=$S/data/contrast13m_200; G=$S/glm
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
miss() { n=0; for m in base32b random32b v4c32b; do r=$(python $Q/remaining.py $m $D/test_eval12.jsonl 1); rm -f $Q/eval/${m}_r*.json; n=$((n+r)); for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $Q/energy/$p/$m/$l.json ] || n=$((n+1)); done; done; done; echo $n; }
echo "START $(date)"
for c in 1 2 3; do m=$(miss); echo "q32 missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/q32eval.sbatch); echo "== q32eval $J"; waitjob $J; done
for m in base32b random32b v4c32b; do cat $Q/eval/test_$m.r*.jsonl | python -c "
import sys,json
seen=set();out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id']);out.append(l)
sys.stdout.write(''.join(out))" > $Q/eval/test_$m.jsonl; mf=$(ls $Q/eval/test_$m.r*.jsonl.manifest.json | head -1); cp $mf $Q/eval/test_$m.jsonl.manifest.json
  contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $Q/eval/test_$m.jsonl --out $Q/eval/report12_$m.json < /dev/null > /dev/null
  python -c "import json;b=json.load(open('$Q/eval/report12_$m.json'))['budgets']['512'];print('Q32 $m n=%d pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f tok=%.0f len_stops=%s'%($(wc -l < $Q/eval/test_$m.jsonl),b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['mean_completion_tokens'],b.get('length_stops')))"; done
contrastforge compare --reports $Q/eval/report12_base32b.json $Q/eval/report12_random32b.json $Q/eval/report12_v4c32b.json --out $Q/eval/cmp.json < /dev/null > /dev/null
python -c "
import json
for x in json.load(open('$Q/eval/cmp.json'))['comparisons']: print('CMP',x['treatment'].split('/')[-1], round(x['mean_paired_group_difference'],3), [round(v,3) for v in x['paired_ci95']])"
echo "== glm smoke"; J=$(submit $L/glmsmoke.sbatch); echo "smoke $J"; waitjob $J; tail -12 $L/cf-glmsmoke-$J.out
for n in random v4c; do tr '\r' '\n' < $G/logs/smoke_$n.err | grep -E "Error|s/it\]|it/s\]|dropping" | tail -3 | cut -c1-200; done
echo "== glm53"; bash $L/glm53_chain.sh
echo "MASTER DONE $(date)"
