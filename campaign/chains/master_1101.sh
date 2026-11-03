S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; Q=$S/q32; G=$S/glm; D=$S/data/contrast13m_200; SD=$S/stdbench
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable "$@" 2>/dev/null) && break; sleep 60; done; echo "$J"; }
done32() { for a in random_s2 random_s3 v4c_s2 v4c_s3; do [ -f $S/runs/sft32b_$a/metrics.json ] || return 1; done; }
doneglm() { for a in $1; do [ -f $S/runs/sftglm_$a/metrics.json ] || return 1; done; }
ck() { for a in $2; do echo "  $1_$a: $(ls $S/runs/$1_$a 2>/dev/null | grep checkpoint | sort -t- -k2 -n | tail -1) $([ -f $S/runs/$1_$a/metrics.json ] && echo DONE)"; done; }
echo "START $(date)"
for c in 1 2 3 4 5; do done32 && break; J=$(submit $L/sft32b_seeds.sbatch); echo "== sft32b seeds round$c $J $(date)"; waitjob $J; ck sft32b "random_s2 random_s3 v4c_s2 v4c_s3"; done
for c in 1 2 3 4 5 6; do doneglm "random_s2 random_s3 v4c_s2 v4c_s3" && break
  for pair in "random_s2 random_s3" "v4c_s2 v4c_s3"; do doneglm "$pair" && continue; J=$(submit --export=ALL,ARMS="$pair" $L/glmsft_seeds.sbatch); echo "== glmsft seeds [$pair] round$c $J $(date)"; waitjob $J; ck sftglm "$pair"; done; done
miss() { local P=$1 n=0; shift; for m in "$@"; do r=$(python $P/remaining.py $m $D/test_eval12.jsonl 1); rm -f $P/eval/${m}_r*.json; n=$((n+r)); for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $P/energy/$p/$m/$l.json ] || n=$((n+1)); done; done; done; echo $n; }
report() { local P=$1 base=$2; shift 2; for m in "$@"; do cat $P/eval/test_$m.r*.jsonl | python -c "
import sys,json
seen=set();out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id']);out.append(l)
sys.stdout.write(''.join(out))" > $P/eval/test_$m.jsonl; cp $(ls $P/eval/test_$m.r*.jsonl.manifest.json | head -1) $P/eval/test_$m.jsonl.manifest.json
  contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $P/eval/test_$m.jsonl --out $P/eval/report12_$m.json < /dev/null > /dev/null
  python -c "import json;b=json.load(open('$P/eval/report12_$m.json'))['budgets']['512'];print('RES $m pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f tok=%.0f len_stops=%s'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['mean_completion_tokens'],b.get('length_stops')))"; done
  contrastforge compare --reports $P/eval/report12_$base.json $(for m in "$@"; do echo $P/eval/report12_$m.json; done) --out $P/eval/cmp_seeds.json < /dev/null > /dev/null
  python -c "
import json
for x in json.load(open('$P/eval/cmp_seeds.json'))['comparisons']: print('CMP',x['treatment'].split('/')[-1], round(x['mean_paired_group_difference'],3), [round(v,3) for v in x['paired_ci95']])"; }
Q32M="random32b_s2 random32b_s3 v4c32b_s2 v4c32b_s3"; GLMM="glmrandom_s2 glmrandom_s3 glmv4c_s2 glmv4c_s3"
for c in 1 2 3; do m=$(miss $Q $Q32M); echo "q32 seeds eval missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/q32eval_seeds.sbatch); echo "== q32eval seeds $J"; waitjob $J; done
report $Q base32b $Q32M
for c in 1 2 3; do n=0; for x in random_s2 random_s3 v4c_s2 v4c_s3; do for p in mc mgsm; do [ -f $SD/out/qwen3-32b-$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break; J=$(submit $L/q32std_seeds.sbatch); echo "== q32std seeds $J"; waitjob $J; done
for c in 1 2 3 4; do m=$(miss $G $GLMM); echo "glm seeds eval missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/glmeval_seeds.sbatch); echo "== glmeval seeds $J"; waitjob $J; done
report $G glmbase $GLMM
for c in 1 2 3; do n=0; for x in random_s2 random_s3 v4c_s2 v4c_s3; do for p in mc mgsm; do [ -f $SD/out/glm47flash-$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break; J=$(submit $L/glmstd_seeds.sbatch); echo "== glmstd seeds $J"; waitjob $J; done
python3 $SD/aggregate.py | grep -E "model|32b|glm47"
echo "MASTER1101 DONE $(date)"
