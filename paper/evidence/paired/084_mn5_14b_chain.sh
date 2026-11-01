S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; L=$S/logs
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
complete14b() { python3 - <<'PY'
import json,os
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D=S+"/contrast13m_200"; ok=True
for k in range(4):
    n=len(json.load(open(f"{D}/mshard{k}.json"))); f=f"{S}/student_base14b.shard{k}.jsonl"
    h=sum(1 for _ in open(f)) if os.path.exists(f) else 0; print(f"shard{k} {h}/{n}"); ok=ok and h>=n
print("COMPLETE" if ok else "INCOMPLETE")
PY
}
# --- probe result decides QoS for long steps
sleep 600
PROBE=$(sacct -j 47200131 -X -o state -n | tr -d ' '); echo "bsccase probe state after 10 min: $PROBE"
if [ "$PROBE" = "COMPLETED" ]; then QOS=acc_bsccase; TL=06:00:00; else QOS=acc_debug; TL=02:00:00; fi; echo "long steps on $QOS"
# --- collect chunks
echo "== wait chunk1 47200130"; waitjob 47200130; complete14b | tail -5
for c in 2 3; do
  if complete14b | grep -q INCOMPLETE; then J=$(sbatch --parsable $L/collect_14b_debug.sbatch); echo "== chunk$c $J $(date)"; waitjob $J; complete14b | tail -5; fi
done
complete14b | grep -q COMPLETE || { echo "collect incomplete, stopping"; exit 1; }
# --- merge + prepare (3 arms)
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"
recs=[json.loads(l) for k in range(4) for l in open(f"{S}/student_base14b.shard{k}.jsonl")]
print("14B records",len(recs),"fingerprints",len({r["run_fingerprint"] for r in recs}),"base acc %.3f"%(sum(r["verification"]["correct"] for r in recs)/len(recs)))
with open(f"{S}/student_base14b.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
json.dump(json.load(open(f"{S}/student_base14b.shard0.jsonl.manifest.json")),open(f"{S}/student_base14b.jsonl.manifest.json","w"),indent=1)
PY
for arm in translated random mined; do
  contrastforge prepare --inputs $D/train.jsonl $D/validation.jsonl --rollouts $S/data/student_base14b.jsonl --strategy $arm --targets worked \
    --tokenizer $S/models/Qwen3-14B --token-budget 3000000 --out $S/data/curricula14b/$arm 2>&1 | grep -E '"training_rows"|unique_training|Error' | tr -d '\n'; echo " <- $arm"
done
python - <<'PY'
import json
base=json.load(open("configs/contrast_sft_mn5.json"))
for arm in ("translated","random","mined"):
    c=dict(base); c["model"]="/gpfs/scratch/bsc21/bsc021885/cf/models/Qwen3-14B"
    c["train_data"]=f"/gpfs/scratch/bsc21/bsc021885/cf/data/curricula14b/{arm}/sft_train.jsonl"; c["validation_data"]=f"/gpfs/scratch/bsc21/bsc021885/cf/data/curricula14b/{arm}/sft_validation.jsonl"
    c["output_dir"]=f"/gpfs/scratch/bsc21/bsc021885/cf/runs/sft14b_{arm}"; json.dump(c,open(f"configs/v14b_{arm}.json","w"),indent=1)
print("14B configs written")
PY
# --- SFT (3 arms, resumable)
mk_sft() { cat > $L/sft14b.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sft14b
#SBATCH --account=bsc21
#SBATCH --qos=$QOS
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:3
#SBATCH --time=$TL
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; cd ~/probeforge; K=0
for arm in translated random mined; do
  R=""; [ -d \$S/runs/sft14b_\$arm ] && ls \$S/runs/sft14b_\$arm | grep -q checkpoint- && R="--resume latest"
  CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sft14b-\$arm.out --error=$L/sft14b-\$arm.err contrastforge train --config configs/v14b_\$arm.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
}
mk_sft
for c in 1 2 3; do
  done_all=1; for arm in translated random mined; do [ -f $S/runs/sft14b_$arm/metrics.json ] || done_all=0; done
  [ $done_all = 1 ] && break
  J=$(sbatch --parsable $L/sft14b.sbatch); echo "== sft14b round$c $J $(date)"; waitjob $J
  for arm in translated random mined; do echo "$arm: $(cat $S/runs/sft14b_$arm/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done
done
# --- eval: 3 adapters × 4 shards (two per GPU sequentially) in one job
cat > $L/eval14b_adapters.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-eval14b-ad
#SBATCH --account=bsc21
#SBATCH --qos=$QOS
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=$TL
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; D=\$S/data/contrast13m_200; T=\$D/test_eval12.jsonl; E=\$S/eval14b; cd ~/probeforge
gpu() { for n in \$2; do CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_\${n}_s\$3.out --error=\$E/collect_\${n}_s\$3.err \\
  contrastforge collect --inputs \$T --only-ids \$D/eshard\$3.json --out \$E/test_\$n.shard\$3.jsonl --model \$S/models/Qwen3-14B --adapter \$S/runs/sft14b_\$n --budgets 512 --samples 1; done; }
gpu 0 "random translated" 0 & gpu 1 "random translated" 1 & gpu 2 "random mined" 2 & gpu 3 "random mined" 3 &
wait
gpu 0 "mined" 0 & gpu 1 "mined" 1 & gpu 2 "translated" 2 & gpu 3 "translated" 3 &
wait
for n in random mined translated; do cat \$E/test_\$n.shard*.jsonl > \$E/test_\$n.jsonl; cp \$E/test_\$n.shard0.jsonl.manifest.json \$E/test_\$n.jsonl.manifest.json; contrastforge evaluate --inputs \$T --rollouts \$E/test_\$n.jsonl --out \$E/report12_\$n.json >/dev/null && echo "report12_\$n ok"; done
contrastforge compare --reports \$E/report12_base14b.json \$E/report12_translated.json \$E/report12_random.json \$E/report12_mined.json --out \$E/comparison12.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/eval14b"
for n in ("base14b","translated","random","mined"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"14B {n:11s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} base_view={b['per_view']['base']:.2f} cf={b['per_view']['counterfactual']:.2f}"); print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())})
for x in json.load(open(f"{E}/comparison12.json"))["comparisons"]: print(x["treatment"].split("/")[-1],round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
echo "end \$(date)"
SB
for c in 1 2; do
  [ -f $S/eval14b/comparison12.json ] && break
  J=$(sbatch --parsable $L/eval14b_adapters.sbatch); echo "== eval14b round$c $J $(date)"; waitjob $J
done
grep -E "^14B|json [0-9.-]+ \[|lang" $L/cf-eval14b-ad-*.out 2>/dev/null | tail -12
echo "CHAIN DONE $(date)"
