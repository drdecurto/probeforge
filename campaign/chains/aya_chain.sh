S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; L=$S/logs; M=$S/models/aya-expanse-8b; E=$S/evalaya; mkdir -p $E
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
# --- preconditions: model downloaded, core60 chain finished (serialise on the single debug slot)
# --- collect base rollouts on the 7,748 mining prompts (resumable debug chunks)
sed -e 's/cf-collect14b/cf-collectaya/' -e "s#Qwen3-14B#aya-expanse-8b#g" -e 's#student_base14b\.shard#student_baseaya.shard#' -e 's#collect-14b-shard#collect-aya-shard#' $L/collect_14b_debug.sbatch > $L/collect_aya_debug.sbatch
grep -E "job-name|phi-4|student_baseaya" $L/collect_aya_debug.sbatch | head -3
complete() { python3 - <<'PY'
import json,os
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D=S+"/contrast13m_200"; ok=True
for k in range(4):
    n=len(json.load(open(f"{D}/mshard{k}.json"))); f=f"{S}/student_baseaya.shard{k}.jsonl"
    h=sum(1 for _ in open(f)) if os.path.exists(f) else 0; print(f"shard{k} {h}/{n}"); ok=ok and h>=n
print("COMPLETE" if ok else "INCOMPLETE")
PY
}
for c in 1 2 3 4; do
  complete | grep -q INCOMPLETE || break
  J=$(submit $L/collect_aya_debug.sbatch); echo "== collect aya chunk$c $J $(date)"; waitjob $J; complete | tail -5
done
complete | grep -q COMPLETE || { echo "collect incomplete, stopping"; exit 1; }
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"
recs=[json.loads(l) for k in range(4) for l in open(f"{S}/student_baseaya.shard{k}.jsonl")]
print("aya records",len(recs),"fingerprints",len({r["run_fingerprint"] for r in recs}),"base acc on mining set %.3f"%(sum(r["verification"]["correct"] for r in recs)/len(recs)),"mean tokens %.0f"%(sum(r["usage"]["completion_tokens"] for r in recs)/len(recs)))
with open(f"{S}/student_baseaya.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
json.dump(json.load(open(f"{S}/student_baseaya.shard0.jsonl.manifest.json")),open(f"{S}/student_baseaya.jsonl.manifest.json","w"),indent=1)
PY
for arm in translated random mined; do
  contrastforge prepare --inputs $D/train.jsonl $D/validation.jsonl --rollouts $S/data/student_baseaya.jsonl --strategy $arm --targets worked \
    --tokenizer $M --token-budget 3000000 --out $S/data/curriculaaya/$arm 2>&1 | grep -E '"training_rows"|unique_training|Error' | tr -d '\n'; echo " <- $arm"
done
python - <<'PY'
import json
base=json.load(open("configs/contrast_sft_mn5.json"))
for arm in ("translated","random","mined"):
    c=dict(base); c["model"]="/gpfs/scratch/bsc21/bsc021885/cf/models/aya-expanse-8b"
    c["train_data"]=f"/gpfs/scratch/bsc21/bsc021885/cf/data/curriculaaya/{arm}/sft_train.jsonl"; c["validation_data"]=f"/gpfs/scratch/bsc21/bsc021885/cf/data/curriculaaya/{arm}/sft_validation.jsonl"
    c["output_dir"]=f"/gpfs/scratch/bsc21/bsc021885/cf/runs/sftaya_{arm}"; json.dump(c,open(f"configs/vaya_{arm}.json","w"),indent=1)
print("aya configs written")
PY
cat > $L/sftaya.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sftaya
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; cd ~/probeforge; K=0
for arm in translated random mined; do
  R=""; [ -d \$S/runs/sftaya_\$arm ] && ls \$S/runs/sftaya_\$arm | grep -q checkpoint- && R="--resume latest"
  [ -f \$S/runs/sftaya_\$arm/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftaya-\$arm.out --error=$L/sftaya-\$arm.err contrastforge train --config configs/vaya_\$arm.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
for c in 1 2 3; do
  done_all=1; for arm in translated random mined; do [ -f $S/runs/sftaya_$arm/metrics.json ] || done_all=0; done
  [ $done_all = 1 ] && break
  J=$(submit $L/sftaya.sbatch); echo "== sftaya round$c $J $(date)"; waitjob $J
  for arm in translated random mined; do echo "$arm: $(cat $S/runs/sftaya_$arm/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done
  tail -3 $L/sftaya-random.err
done
# --- eval on test_eval12 (1,248 prompts, 4 shards): job A = base + random, job B = mined + translated
mkeval() { cat > $L/evalaya_$1.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-evalaya-$1
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
source ~/probeforge/slurm/environment.sh; S=$S; D=\$S/data/contrast13m_200; T=\$D/test_eval12.jsonl; E=$E; M=$M; cd ~/probeforge
gpu() { for n in \$2; do A=""; [ \$n != base ] && A="--adapter \$S/runs/sftaya_\$n"; [ -f \$E/test_\$n.shard\$3.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_\${n}_s\$3.out --error=\$E/collect_\${n}_s\$3.err \\
  contrastforge collect --inputs \$T --only-ids \$D/eshard\$3.json --out \$E/test_\$n.shard\$3.jsonl --model \$M \$A --budgets 512 --samples 1; done; }
gpu 0 "$2" 0 & gpu 1 "$2" 1 & gpu 2 "$2" 2 & gpu 3 "$2" 3 &
wait; echo "end \$(date)"
SB
}
mkeval A "base random"; mkeval B "mined translated"
for half in A B; do
  [ $half = A ] && names="base random" || names="mined translated"
  for c in 1 2; do
    ok=1; for n in $names; do for sh in 0 1 2 3; do [ -f $E/test_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; done; [ $ok = 1 ] && break
    J=$(submit $L/evalaya_$half.sbatch); echo "== evalaya $half round$c $J $(date)"; waitjob $J
  done
done
T=$D/test_eval12.jsonl
for n in base random mined translated; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok ($(wc -l < $E/test_$n.jsonl) rows)"; done
contrastforge compare --reports $E/report12_base.json $E/report12_translated.json $E/report12_random.json $E/report12_mined.json --out $E/comparison12.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalaya"
for n in ("base","translated","random","mined"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"AYA {n:11s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} base_view={b['per_view']['base']:.2f} cf={b['per_view']['counterfactual']:.2f} tok={b['mean_completion_tokens']:.0f}"); print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())})
for x in json.load(open(f"{E}/comparison12.json"))["comparisons"]: print(x["treatment"].split("/")[-1],round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
echo "CHAIN DONE $(date)"
