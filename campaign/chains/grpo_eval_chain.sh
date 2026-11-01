S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; B=$S/energybench; SD=$S/stdbench; E=$S/evalgrpo; D=$S/data/contrast13m_200
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"; for i in $(seq 1 1440); do grep -q "CHAIN DONE" $L/grpo8b_chain.log 2>/dev/null && break; sleep 60; done
[ -f $S/runs/grpo_8b/adapter_model.safetensors ] || { echo "no grpo adapter"; exit 1; }
for c in 1 2; do ok=1; for sh in 0 1 2 3; do [ -f $E/test_grpo8b.shard$sh.jsonl.manifest.json ] || ok=0; done; [ $ok = 1 ] && break; J=$(submit $L/evalgrpo.sbatch); echo "== evalgrpo $J"; waitjob $J; done
cat $E/test_grpo8b.shard*.jsonl > $E/test_grpo8b.jsonl; cp $E/test_grpo8b.shard0.jsonl.manifest.json $E/test_grpo8b.jsonl.manifest.json
contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $E/test_grpo8b.jsonl --out $E/report12_grpo8b.json < /dev/null > /dev/null
contrastforge compare --reports $S/evalv4c/report12_8b_s1.json $E/report12_grpo8b.json --out $E/cmp_grpo_vs_v4c.json < /dev/null > /dev/null
python -c "
import json;b=json.load(open('$E/report12_grpo8b.json'))['budgets']['512'];print('GRPO8B pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['mean_completion_tokens']))
x=json.load(open('$E/cmp_grpo_vs_v4c.json'))['comparisons'][0];print('GRPO - v4c_s1', round(x['mean_paired_group_difference'],3), [round(v,3) for v in x['paired_ci95']])"
miss() { n=0; for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do for o in out out_chat; do [ -s $B/$o/grpo-8b/$l.json ] || n=$((n+1)); done; done; echo $n; }
for c in 1 2; do m=$(miss); echo "energy grpo missing $m"; [ $m = 0 ] && break; J=$(submit $L/energy_grpo.sbatch); echo "== energygrpo $J"; waitjob $J; done
python3 $B/answer_acc.py | grep -E "grpo-8b|v4c-8b_s1|qwen3-8b-base"
for c in 1 2; do n=0; for p in mc mgsm; do [ -f $SD/out/grpo-8b/$p/DONE ] || n=$((n+1)); done; [ $n = 0 ] && break; J=$(submit $L/stdgrpo.sbatch); echo "== stdgrpo $J"; waitjob $J; done
python3 $SD/aggregate.py | grep -E "model|grpo|v4c-8b_s1|qwen3-8b-base"
echo "CHAIN DONE $(date)"
