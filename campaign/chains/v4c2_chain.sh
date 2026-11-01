S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; V=$S/v4; B=$S/energybench; SD=$S/stdbench; E=$S/evalv4c
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
# A. data
: data ready
grep -h "^[a-z-]* {" $V/gsm_c/log_gpu*.txt; wc -l $V/gsm_c/gsm_*.jsonl | tail -1
# B/C. mixtures + configs
while IFS='|' read name base cur cfg seed; do
  [ -f $V/mixc_$name/sft_train.jsonl ] || python $V/v4_mix.py --contrast $S/data/$cur --gsm $V/gsm_c --tokenizer $S/models/$base --out $V/mixc_$name --seed $seed | sed "s/^/$name /"
  python - "$name" "$cfg" "$seed" "$base" <<'PY'
import json,sys
name,cfg,seed=sys.argv[1:4]; c=json.load(open(cfg)); V="/gpfs/scratch/bsc21/bsc021885/cf/v4"
c.update(model=f"/gpfs/scratch/bsc21/bsc021885/cf/models/{sys.argv[4]}", train_data=f"{V}/mixc_{name}/sft_train.jsonl", validation_data=f"{V}/mixc_{name}/sft_validation.jsonl", output_dir=f"/gpfs/scratch/bsc21/bsc021885/cf/runs/sftv4c_{name}", seed=int(seed))
json.dump(c,open(f"configs/v4c_{name}.json","w"),indent=1)
PY
done < $V/v4c_runs2.txt
# D. SFT
alltrained() { while IFS='|' read name x; do [ -f $S/runs/sftv4c_$name/metrics.json ] || return 1; done < $V/v4c_runs2.txt; }
for c in 1 2 3; do alltrained && break; J=$(submit $L/sftv4c2.sbatch); echo "== sftv4c round$c $J $(date)"; waitjob $J; done
while IFS='|' read name x; do echo "$name $(tr -d '\n ' < $S/runs/sftv4c_$name/metrics.json 2>/dev/null | grep -o '"eval_loss":[0-9.]*')"; done < $V/v4c_runs2.txt
# E. main benchmark
alleval() { while IFS='|' read name x; do [ -f $E/test_$name.jsonl.manifest.json ] || return 1; done < $V/v4c_runs2.txt; }
for c in 1 2; do alleval && break; J=$(submit $L/evalv4c2.sbatch); echo "== evalv4c round$c $J $(date)"; waitjob $J; done
while IFS='|' read name x; do contrastforge evaluate --inputs $S/data/contrast13m_200/test_eval12.jsonl --rollouts $E/test_$name.jsonl --out $E/report12_$name.json >/dev/null
  python -c "import json;b=json.load(open('$E/report12_$name.json'))['budgets']['512'];print('V4C $name pass@1=%.3f worst=%.3f boundary=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['mean_completion_tokens']))"; done < $V/v4c_runs2.txt
# F. energy
miss() { n=0; while IFS='|' read s b a; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do for o in out out_chat; do [ -s $B/$o/$s/$l.json ] || n=$((n+1)); done; done; done < $B/slots_v4c2.txt; echo $n; }
for c in 1 2 3 4; do m=$(miss); echo "energy v4c missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/energy_v4c2.sbatch); echo "== energyv4c $J"; waitjob $J; done
python3 $B/answer_acc.py | grep -E "v4c-14b|v4c-aya|aya-expanse|qwen3-14b-base"
# G. merge + stdbench
for c in 1 2; do n=0; while IFS='|' read name x; do [ -f $S/models/merged/v4c-$name/config.json ] || n=$((n+1)); done < $V/v4c_runs2.txt; [ $n = 0 ] && break
  J=$(submit $L/stdmerge_v4c2.sbatch); echo "== stdmerge_v4c $J"; waitjob $J; grep merge $L/stdmergev4c-2-$J.out; done
for c in 1 2 3; do n=0; while IFS='|' read name x; do for p in mc mgsm; do [ -f $SD/out/v4c-$name/$p/DONE ] || n=$((n+1)); done; done < $V/v4c_runs2.txt; [ $n = 0 ] && break
  J=$(submit $L/stdv4c2.sbatch); echo "== stdv4c $J"; waitjob $J; grep -E " ok | FAIL" $L/stdv4c2-$J.out | tail -16; done
python3 $SD/aggregate.py | grep -E "model|v4c-14b|v4c-aya|aya-expanse|qwen3-14b-base"
echo "CHAIN DONE $(date)"
contrastforge evaluate --inputs $S/data/contrast13m_200/test_eval12.jsonl --rollouts $E/test_aya_base.jsonl --out $E/report12_aya_base.json < /dev/null >/dev/null && python -c "import json;b=json.load(open('$E/report12_aya_base.json'))['budgets']['512'];print('AYA BASE pass@1=%.3f worst=%.3f boundary=%.2f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary']))"
