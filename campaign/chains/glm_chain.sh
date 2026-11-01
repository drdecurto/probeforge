S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; G=$S/glm; D=$S/data/contrast13m_200
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
for i in $(seq 1 1440); do [ -f $S/runs/glmsmoke_random/metrics.json ] && [ -f $S/runs/glmsmoke_v4c/metrics.json ] && break; grep -q "MASTER DONE\|== glm53" $L/master_1029.log && ! [ -f $S/runs/glmsmoke_random/metrics.json ] && { echo "smoke failed; stopping"; exit 1; }; sleep 60; done
echo "smoke ok $(date)"; for n in random v4c; do tr '\r' '\n' < $G/logs/smoke_$n.err | grep -E "s/it\]|it/s\]" | tail -1 | cut -c1-120; done
for c in $(seq 1 8); do [ -f $S/runs/sftglm_random/metrics.json ] && [ -f $S/runs/sftglm_v4c/metrics.json ] && break; J=$(submit $L/glmsft.sbatch); echo "== glmsft round$c $J $(date)"; waitjob $J
  for n in random v4c; do echo "  $n: $(ls $S/runs/sftglm_$n 2>/dev/null | grep checkpoint | sort -t- -k2 -n | tail -1) $([ -f $S/runs/sftglm_$n/metrics.json ] && echo DONE)"; done; done
miss() { n=0; for m in glmbase glmrandom glmv4c; do r=$(python $G/remaining.py $m $D/test_eval12.jsonl 1); rm -f $G/eval/${m}_r*.json; n=$((n+r)); for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $G/energy/$p/$m/$l.json ] || n=$((n+1)); done; done; done; echo $n; }
for c in 1 2 3 4; do m=$(miss); echo "glm eval missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/glmeval.sbatch); echo "== glmeval $J"; waitjob $J; grep -E "merge|remaining|done|failed" $L/glm-eval-$J.out 2>/dev/null | head; done
for m in glmbase glmrandom glmv4c; do cat $G/eval/test_$m.r*.jsonl | python -c "
import sys,json
seen=set();out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id']);out.append(l)
sys.stdout.write(''.join(out))" > $G/eval/test_$m.jsonl; cp $(ls $G/eval/test_$m.r*.jsonl.manifest.json | head -1) $G/eval/test_$m.jsonl.manifest.json
  contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $G/eval/test_$m.jsonl --out $G/eval/report12_$m.json < /dev/null > /dev/null
  python -c "import json;b=json.load(open('$G/eval/report12_$m.json'))['budgets']['512'];print('GLM $m pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f tok=%.0f len_stops=%s'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['mean_completion_tokens'],b.get('length_stops')))"; done
contrastforge compare --reports $G/eval/report12_glmbase.json $G/eval/report12_glmrandom.json $G/eval/report12_glmv4c.json --out $G/eval/cmp.json < /dev/null > /dev/null
python -c "
import json
for x in json.load(open('$G/eval/cmp.json'))['comparisons']: print('CMP',x['treatment'].split('/')[-1], round(x['mean_paired_group_difference'],3), [round(v,3) for v in x['paired_ci95']])"
for c in 1 2 3; do n=0; for x in glm47flash-base glm47flash-random glm47flash-v4c; do for p in mc mgsm; do [ -f $S/stdbench/out/$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break; J=$(submit $L/glmstd.sbatch); echo "== glmstd $J"; waitjob $J; grep -E " ok | FAIL" $L/glm-std-$J.out; done
python3 $S/stdbench/aggregate.py | grep -E "model|glm47"
echo "GLM CHAIN DONE $(date)"
