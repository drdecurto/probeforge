S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; Q=$S/q32; SD=$S/stdbench; D=$S/data/contrast13m_200
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
for c in 1 2 3 4 5; do [ -f $S/runs/sft32b_random/metrics.json ] && [ -f $S/runs/sft32b_v4c/metrics.json ] && break; J=$(submit $L/sft32b.sbatch); echo "== sft32b round$c $J $(date)"; waitjob $J
  for n in random v4c; do echo "  $n: $(ls $S/runs/sft32b_$n 2>/dev/null | grep checkpoint | tail -1) $(tr -d '\n ' < $S/runs/sft32b_$n/metrics.json 2>/dev/null | grep -o '"eval_loss":[0-9.]*')"; done; done
emiss() { n=0; for m in base32b random32b v4c32b; do [ -f $Q/eval/test_$m.jsonl.manifest.json ] || n=$((n+1)); for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $Q/energy/$p/$m/$l.json ] || n=$((n+1)); done; done; done; echo $n; }
for c in 1 2 3; do m=$(emiss); echo "q32 eval missing $m"; [ $m = 0 ] && break; J=$(submit $L/q32eval.sbatch); echo "== q32eval $J $(date)"; waitjob $J; done
for m in base32b random32b v4c32b; do contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $Q/eval/test_$m.jsonl --out $Q/eval/report12_$m.json < /dev/null > /dev/null
  python -c "import json;b=json.load(open('$Q/eval/report12_$m.json'))['budgets']['512'];print('Q32 $m pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['mean_completion_tokens']))"; done
contrastforge compare --reports $Q/eval/report12_base32b.json $Q/eval/report12_random32b.json $Q/eval/report12_v4c32b.json --out $Q/eval/cmp.json < /dev/null > /dev/null
python -c "
import json
for x in json.load(open('$Q/eval/cmp.json'))['comparisons']: print(x['treatment'].split('/')[-1], round(x['mean_paired_group_difference'],3), [round(v,3) for v in x['paired_ci95']])"
for c in 1 2 3; do n=0; for x in qwen3-32b-base qwen3-32b-random qwen3-32b-v4c; do for p in mc mgsm; do [ -f $SD/out/$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break; J=$(submit $L/q32std.sbatch); echo "== q32std $J"; waitjob $J; grep -E "merge| ok | FAIL" $L/q32-std-$J.out; done
python3 $SD/aggregate.py | grep -E "model|32b|qwen3-14b-base"
echo "CHAIN DONE $(date)"
