S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; K=$S/kolibri; D=$S/data/contrast13m_200; SD=$S/stdbench
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable "$@" 2>/dev/null) && break; sleep 60; done; echo "$J"; }
ad() { local d=$S/runs/sftkol_$1; [ -f $d/metrics.json ] && [ -f $d/adapter_model.safetensors ] && echo $d; }
miss() { local m=$1 n; n=$(python $K/remaining.py $m $D/test_eval12.jsonl 1); rm -f $K/eval/${m}_r*.json
  for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $K/energy/$p/$m/$l.json ] || n=$((n+1)); done; done
  for part in mc mgsm; do [ -f $SD/out/kol-$m/$part/DONE ] || n=$((n+1)); done; echo $n; }
echo "START $(date)"
for trio in "random_s1 random_s2 random_s3" "v4c_s1 v4c_s2 v4c_s3"; do
  for i in $(seq 1 600); do ok=1; for a in $trio; do [ -n "$(ad $a)" ] || ok=0; done; [ $ok = 1 ] && break; sleep 60; done; echo "trio [$trio] trained $(date)"
  P=""; for a in $trio; do P="$P $(ad $a)|$S/models/merged/kol_$a"; done
  for c in 1 2; do n=0; for a in $trio; do [ -f $S/models/merged/kol_$a/.merged ] || n=1; done; [ $n = 0 ] && break; J=$(submit --export=ALL,PAIRS="$P" $L/kolmerge.sbatch); echo "== merge $J"; waitjob $J; done
  SP=""; for a in $trio; do SP="$SP kol$a|$S/models/merged/kol_$a"; done
  for c in 1 2 3 4; do n=0; for a in $trio; do n=$((n+$(miss kol$a))); done; echo "eval missing $n $(date +%T)"; [ $n = 0 ] && break; J=$(submit --export=ALL,KSPECS3="$SP" $L/koleval3.sbatch); echo "== koleval3 [$trio] $J $(date)"; waitjob $J; done
  for a in $trio; do m=kol$a
    cat $K/eval/test_$m.r*.jsonl | python -c "
import sys,json
seen=set();out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id']);out.append(l)
sys.stdout.write(''.join(out))" > $K/eval/test_$m.jsonl; cp $(ls $K/eval/test_$m.r*.jsonl.manifest.json | head -1) $K/eval/test_$m.jsonl.manifest.json
    contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $K/eval/test_$m.jsonl --out $K/eval/report12_$m.json < /dev/null > /dev/null
    python -c "import json;b=json.load(open('$K/eval/report12_$m.json'))['budgets']['512'];print('RES $m pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f base_view=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['per_view']['base'],b['mean_completion_tokens']))"
  done
done
contrastforge compare --reports $K/eval/report12_kolbase.json $(for a in random_s1 random_s2 random_s3 v4c_s1 v4c_s2 v4c_s3; do echo $K/eval/report12_kol$a.json; done) --out $K/eval/cmp_kol.json < /dev/null > /dev/null
python -c "
import json
for x in json.load(open('$K/eval/cmp_kol.json'))['comparisons']: print('CMP',x['treatment'].split('/')[-1], round(x['mean_paired_group_difference'],3), [round(v,3) for v in x['paired_ci95']])"
echo "KOLPOST DONE $(date)"
