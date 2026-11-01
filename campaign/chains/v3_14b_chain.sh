S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; D3=$S/data/contrast13s_200; D=$S/data/contrast13m_200; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13
M=$S/models/Qwen3-14B; E=$S/eval14b; E6=$S/evalpacks/diag60
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; [ -f "$1" ] || { echo "MISSING sbatch file $1" >&2; return 1; }
  for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
echo "START $(date)"
# ---------- all sbatch files up front
sed -e 's/cf-collect-v3new/cf-collect-v3n14b/' -e 's#Qwen3-8B#Qwen3-14B#g' -e 's#student_base_v3new\.shard#student_base14b_v3new.shard#g' -e 's#collect-v3new-shard#collect-v3n14b-shard#g' -e 's/--time=01:00:00/--time=02:00:00/' $L/collect_v3new.sbatch > $L/collect_v3n14b.sbatch
cat > $L/sftv3_14b.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv3-14b
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
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; cd ~/probeforge; K=0
for arm in random mined; do
  R=""; [ -d \$S/runs/sftv3_14b_\$arm ] && ls \$S/runs/sftv3_14b_\$arm | grep -q checkpoint- && R="--resume latest"
  [ -f \$S/runs/sftv3_14b_\$arm/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftv3_14b-\$arm.out --error=$L/sftv3_14b-\$arm.err contrastforge train --config configs/v3_14b_\$arm.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
cat > $L/evalv3_14b.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv3-14b
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; D=$D; T=\$D/test_eval12.jsonl; E=$E; M=$M; cd ~/probeforge
gpu() { for n in \$2; do [ -f \$E/test_v3_\$n.shard\$3.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_v3\${n}_s\$3.out --error=\$E/collect_v3\${n}_s\$3.err \\
  contrastforge collect --inputs \$T --only-ids \$D/eshard\$3.json --out \$E/test_v3_\$n.shard\$3.jsonl --model \$M --adapter \$S/runs/sftv3_14b_\$n --budgets 512 --samples 1; done; }
gpu 0 "random mined" 0 & gpu 1 "random mined" 1 & gpu 2 "random mined" 2 & gpu 3 "random mined" 3 &
wait; echo "end \$(date)"
SB
mk() { cat > $L/$1.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=$1
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; P=$P; C=$C; E=$E6; cd ~/probeforge
A=""; [ -n "$4" ] && A="--adapter $4"; K=0
for sh in $2; do CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_$5_s\$sh.out --error=\$E/collect_$5_s\$sh.err \\
  contrastforge collect --inputs \$C/test_diagnostic60.jsonl --only-ids \$P/diag60_shard\$sh.json --out \$E/test_$5.shard\$sh.jsonl --model $3 \$A --budgets 512 --samples 1 & K=\$((K+1)); done; wait
echo "end \$(date)"
SB
}
mk d60-v3random14b-A "0 1 2 3" $M "$S/runs/sftv3_14b_random" v3random14b; mk d60-v3random14b-B "4 5 6 7" $M "$S/runs/sftv3_14b_random" v3random14b
ls $L/collect_v3n14b.sbatch $L/sftv3_14b.sbatch $L/evalv3_14b.sbatch $L/d60-v3random14b-?.sbatch
# ---------- A) collect the new-view prompts with Qwen3-14B (resumable)
complete() { python3 - <<'PY'
import json,os
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D3=S+"/contrast13s_200"; ok=True
for k in range(4):
    n=len(json.load(open(f"{D3}/nshard{k}.json"))); f=f"{S}/student_base14b_v3new.shard{k}.jsonl"
    h=sum(1 for _ in open(f)) if os.path.exists(f) else 0; print(f"shard{k} {h}/{n}"); ok=ok and h>=n
print("COMPLETE" if ok else "INCOMPLETE")
PY
}
for c in 1 2; do complete | grep -q INCOMPLETE || break; J=$(submit $L/collect_v3n14b.sbatch); echo "== collect v3new 14B chunk$c $J $(date)"; waitjob $J; complete | tail -1; done
complete | grep -q COMPLETE || { echo "collect incomplete, stopping"; exit 1; }
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D3=S+"/contrast13s_200"
new=[json.loads(l) for k in range(4) for l in open(f"{S}/student_base14b_v3new.shard{k}.jsonl")]
fp=new[0]["run_fingerprint"]
old=[json.loads(l) for l in open(f"{S}/student_base14b.jsonl")]
for r in old: r["run_fingerprint"]=fp
ids3=set(json.load(open(f"{D3}/mining_ids.json")))
allr=[r for r in old+new if r["id"] in ids3]
with open(f"{S}/student_base14b_v3.jsonl","w") as f:
    for r in allr: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
m=json.load(open(f"{S}/student_base14b_v3new.shard0.jsonl.manifest.json")); m["note"]="14B v2 rollouts (identical rows) + new boundary_alt/infeasible views"; json.dump(m,open(f"{S}/student_base14b_v3.jsonl.manifest.json","w"),indent=1)
print("14B v3 rollouts",len(allr),"new",len(new),"14B base acc on new views %.3f"%(sum(r["verification"]["correct"] for r in new)/len(new)))
PY
# ---------- B) prepare + configs
for arm in random mined; do
  contrastforge prepare --inputs $D3/train.jsonl $D3/validation.jsonl --rollouts $S/data/student_base14b_v3.jsonl --strategy $arm --targets worked \
    --tokenizer $M --token-budget 3000000 --out $S/data/curriculav3_14b/$arm 2>&1 | grep -E '"training_rows"|unique_training|Error' | tr -d '\n'; echo " <- $arm"
done
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf"
for arm in ("random","mined"):
    c=json.load(open("configs/contrast_sft_mn5.json"))
    c.update(model=f"{S}/models/Qwen3-14B",train_data=f"{S}/data/curriculav3_14b/{arm}/sft_train.jsonl",validation_data=f"{S}/data/curriculav3_14b/{arm}/sft_validation.jsonl",output_dir=f"{S}/runs/sftv3_14b_{arm}",seed=1)
    json.dump(c,open(f"configs/v3_14b_{arm}.json","w"),indent=1)
print("v3 14B configs written (seed 1)")
PY
# ---------- C) SFT (2 arms, resumable)
for c in 1 2 3; do
  ok=1; for arm in random mined; do [ -f $S/runs/sftv3_14b_$arm/metrics.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/sftv3_14b.sbatch); echo "== sftv3 14B round$c $J $(date)"; waitjob $J
  for arm in random mined; do echo "$arm: $(cat $S/runs/sftv3_14b_$arm/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done
done
# ---------- D) eval on test_eval12
for c in 1 2; do
  ok=1; for n in random mined; do for sh in 0 1 2 3; do [ -f $E/test_v3_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; done; [ $ok = 1 ] && break
  J=$(submit $L/evalv3_14b.sbatch); echo "== evalv3 14B round$c $J $(date)"; waitjob $J
done
T=$D/test_eval12.jsonl
for n in random mined; do cat $E/test_v3_$n.shard*.jsonl > $E/test_v3_$n.jsonl; cp $E/test_v3_$n.shard0.jsonl.manifest.json $E/test_v3_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_v3_$n.jsonl --out $E/report12_v3_$n.json >/dev/null && echo "report12_v3_$n ok ($(wc -l < $E/test_v3_$n.jsonl) rows)"; done
contrastforge compare --reports $E/report12_random.json $E/report12_v3_random.json --out $E/comparison12_v3_random.json >/dev/null
contrastforge compare --reports $E/report12_mined.json $E/report12_v3_mined.json --out $E/comparison12_v3_mined.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/eval14b"
for n in ("base14b","random","mined","v3_random","v3_mined"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"V3-14B {n:10s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} view={ {k:round(v,2) for k,v in b['per_view'].items()} } tok={b['mean_completion_tokens']:.0f}")
for f in ("comparison12_v3_random","comparison12_v3_mined"):
    for x in json.load(open(f"{E}/{f}.json"))["comparisons"]: print(f,"v3 vs v2:",round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
# ---------- E) diag60 with the v3 random 14B adapter (boundary under the alternate construction)
need() { for sh in $2; do f=$E6/test_$1.shard$sh.jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 780 ] || return 0; done; return 1; }
for half in A B; do [ $half = A ] && shards="0 1 2 3" || shards="4 5 6 7"
  if need v3random14b "$shards"; then J=$(submit $L/d60-v3random14b-$half.sbatch); echo "== d60-v3random14b-$half $J $(date)"; waitjob $J; else echo "== d60-v3random14b-$half already complete"; fi; done
cat $E6/test_v3random14b.shard*.jsonl > $E6/test_v3random14b.jsonl; cp $E6/test_v3random14b.shard0.jsonl.manifest.json $E6/test_v3random14b.jsonl.manifest.json
contrastforge evaluate --inputs $C/test_diagnostic60.jsonl --rollouts $E6/test_v3random14b.jsonl --out $E6/report_v3random14b.json >/dev/null && echo "report_v3random14b ok"
contrastforge compare --reports $E6/report_random14b.json $E6/report_v3random14b.json --out $E6/comparison_v3_14b.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalpacks/diag60"
for n in ("random14b","v3random14b"):
    b=json.load(open(f"{E}/report_{n}.json"))["budgets"]["512"]
    print(f"DIAG60 {n:12s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} ci={[round(x,3) for x in b['group_mean_accuracy_ci95']]}"); print("   view",{k:round(v,2) for k,v in b["per_view"].items()}); print("   family",{k:round(v,2) for k,v in b["per_family"].items()})
for x in json.load(open(f"{E}/comparison_v3_14b.json"))["comparisons"]: print("diag60 v3 vs v2 (14B):",round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
echo "CHAIN DONE $(date)"
