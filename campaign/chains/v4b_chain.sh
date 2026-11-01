S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; V=$S/v4; B=$S/energybench; SD=$S/stdbench
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
for i in $(seq 1 2880); do [ -f $V/gsm_b/REGEN_DONE ] && break; sleep 60; done
wc -l $V/gsm_b/gsm_*.jsonl | tail -14; grep -h "^[a-z-]* {" $V/gsm_b/log_gpu*.txt
for i in $(seq 1 2880); do grep -q "CHAIN DONE" $L/phi4mini_chain.log 2>/dev/null && break; sleep 60; done
echo "inputs ready $(date)"
# --- mixtures
python $V/v4_mix.py --contrast $S/data/curriculav2/mined --gsm $V/gsm_b --tokenizer $S/models/Qwen3-8B --out $V/mixb_8b
python $V/v4_mix.py --contrast $S/data/curriculaphi4mini/mined --gsm $V/gsm_b --tokenizer $S/models/Phi-4-mini-instruct --out $V/mixb_phi4mini
python - <<'PY'
import json
V="/gpfs/scratch/bsc21/bsc021885/cf/v4"
for name, base in (("8b","configs/v2_mined_s1.json"),("phi4mini","configs/vphi4mini_mined.json")):
    c=json.load(open(base)); c["train_data"]=f"{V}/mixb_{name}/sft_train.jsonl"; c["validation_data"]=f"{V}/mixb_{name}/sft_validation.jsonl"
    c["output_dir"]=f"/gpfs/scratch/bsc21/bsc021885/cf/runs/sftv4b_{name}"; c["seed"]=1; json.dump(c,open(f"configs/v4b_{name}.json","w"),indent=1); print(name, c["model"], c.get("target_modules"))
PY
cat > $L/sftv4b.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv4b
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
source ~/probeforge/slurm/environment.sh; cd ~/probeforge; K=0
for n in 8b phi4mini; do R=""; ls $S/runs/sftv4b_\$n 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftv4b_\$n/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftv4b-\$n.out --error=$L/sftv4b-\$n.err contrastforge train --config configs/v4b_\$n.json \$R &
  K=\$((K+1)); done; wait; echo "end \$(date)"
SB
for c in 1 2 3; do [ -f $S/runs/sftv4b_8b/metrics.json ] && [ -f $S/runs/sftv4b_phi4mini/metrics.json ] && break
  J=$(submit $L/sftv4b.sbatch); echo "== sftv4b round$c $J $(date)"; waitjob $J; for n in 8b phi4mini; do echo "$n $(tr -d '\n ' < $S/runs/sftv4b_$n/metrics.json 2>/dev/null | grep -o '"eval_loss":[0-9.]*')"; done; done
# --- main benchmark (test_eval12), 8 shards over 8 GPUs (2 nodes x 4)
E=$S/evalv4b; mkdir -p $E; D=$S/data/contrast13m_200
cat > $L/evalv4b.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv4b
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=$L/%x-%j.out
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
gpu() { n=\$1; m=\$2; sh=\$3; [ -f $E/test_\$n.shard\$sh.jsonl.manifest.json ] && return
  CUDA_VISIBLE_DEVICES=\$4 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_\${n}_s\$sh.out --error=$E/collect_\${n}_s\$sh.err \\
  contrastforge collect --inputs $D/test_eval12.jsonl --only-ids $D/eshard\$sh.json --out $E/test_\$n.shard\$sh.jsonl --model \$m --adapter $S/runs/sftv4b_\$n --budgets 512 --samples 1; }
for sh in 0 1 2 3; do ( gpu 8b $S/models/Qwen3-8B \$sh \$sh; gpu phi4mini $S/models/Phi-4-mini-instruct \$sh \$sh ) & done; wait
SB
for c in 1 2; do ok=1; for n in 8b phi4mini; do for sh in 0 1 2 3; do [ -f $E/test_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; done; [ $ok = 1 ] && break
  J=$(submit $L/evalv4b.sbatch); echo "== evalv4b round$c $J $(date)"; waitjob $J; done
for n in 8b phi4mini; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json
  contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $E/test_$n.jsonl --out $E/report12_v4b_$n.json >/dev/null
  python -c "import json;b=json.load(open('$E/report12_v4b_$n.json'))['budgets']['512'];print('V4B $n pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['mean_completion_tokens']))"; done
# --- energy benchmark raw + chat (2 nodes) and stdbench (merged)
printf 'qwen3-8b-v4b|Qwen3-8B|sftv4b_8b\nphi-4-mini-v4b|Phi-4-mini-instruct|sftv4b_phi4mini\n' > $B/slots_v4b.txt
sed -e 's#--job-name=energyp4m#--job-name=energyv4b#' -e 's#slots_phi4mini.txt#slots_v4b.txt#g' $L/energy_phi4mini.sbatch > $L/energy_v4b.sbatch
miss() { n=0; while IFS='|' read s b a; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do for o in out out_chat; do [ -s $B/$o/$s/$l.json ] || n=$((n+1)); done; done; done < $B/slots_v4b.txt; echo $n; }
for c in 1 2 3; do m=$(miss); echo "energy v4 missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/energy_v4b.sbatch); echo "== energyv4b $J"; waitjob $J; done
sed -e 's#phi-4-mini-\$a#v4b-$a#g' -e 's#for a in mined random#for a in 8b phi4mini#' -e 's#\$S/models/Phi-4-mini-instruct \$S/runs/sftphi4mini_\$a#$([ $a = 8b ] \&\& echo $S/models/Qwen3-8B || echo $S/models/Phi-4-mini-instruct) $S/runs/sftv4b_$a#' $SD/merge_p4m.sh > $SD/merge_v4b.sh; chmod +x $SD/merge_v4b.sh; grep -n "merge.py" $SD/merge_v4b.sh
grep -q "^qwen3-8b-v4b|" $SD/models.txt || printf 'qwen3-8b-v4b|merged/v4b-8b|\nphi-4-mini-v4b|merged/v4b-phi4mini|\n' >> $SD/models.txt
sed -e 's#--job-name=stdp4m#--job-name=stdv4b#' -e 's#merge_p4m.sh#merge_v4b.sh#' $L/stdp4m.sbatch > $L/stdv4b.sbatch
for c in 1 2; do n=0; for x in qwen3-8b-v4b phi-4-mini-v4b; do for p in mc mgsm; do [ -f $SD/out/$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break
  J=$(submit $L/stdv4b.sbatch); echo "== stdv4b $J"; waitjob $J; grep -E "merge| ok | FAIL" $L/stdv4b-$J.out | tail; done
python3 $SD/aggregate.py | grep -E "model|v4|qwen3-8b|phi-4-mini"
echo "CHAIN DONE $(date)"
