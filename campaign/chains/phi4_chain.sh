S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; L=$S/logs; M=$S/models/phi-4; E=$S/evalphi4; mkdir -p $E
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
# --- preconditions: model downloaded, core60 chain finished (serialise on the single debug slot)
for i in $(seq 1 240); do grep -q "^DONE" $L/phi4_download.log 2>/dev/null && break; sleep 60; done
grep -q "^DONE" $L/phi4_download.log || { echo "phi-4 download not finished, stopping"; exit 1; }
grep -E "template OK|eos:" $L/phi4_download.log
for i in $(seq 1 1440); do grep -q "CHAIN DONE" $L/core60_chain.log 2>/dev/null && break; sleep 60; done
echo "core60 chain done, starting phi-4 at $(date)"
# --- collect base rollouts on the 7,748 mining prompts (resumable debug chunks)
sed -e 's/cf-collect14b/cf-collectphi4/' -e "s#$S/models/Qwen3-14B#$M#" -e 's#student_base14b\.shard#student_basephi4.shard#' -e 's#collect-14b-shard#collect-phi4-shard#' $L/collect_14b_debug.sbatch > $L/collect_phi4_debug.sbatch
grep -E "job-name|phi-4|student_basephi4" $L/collect_phi4_debug.sbatch | head -3
complete() { python3 - <<'PY'
import json,os
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D=S+"/contrast13m_200"; ok=True
for k in range(4):
    n=len(json.load(open(f"{D}/mshard{k}.json"))); f=f"{S}/student_basephi4.shard{k}.jsonl"
    h=sum(1 for _ in open(f)) if os.path.exists(f) else 0; print(f"shard{k} {h}/{n}"); ok=ok and h>=n
print("COMPLETE" if ok else "INCOMPLETE")
PY
}
for c in 1 2 3 4; do
  complete | grep -q INCOMPLETE || break
  J=$(submit $L/collect_phi4_debug.sbatch); echo "== collect phi4 chunk$c $J $(date)"; waitjob $J; complete | tail -5
done
complete | grep -q COMPLETE || { echo "collect incomplete, stopping"; exit 1; }
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"
recs=[json.loads(l) for k in range(4) for l in open(f"{S}/student_basephi4.shard{k}.jsonl")]
print("phi4 records",len(recs),"fingerprints",len({r["run_fingerprint"] for r in recs}),"base acc on mining set %.3f"%(sum(r["verification"]["correct"] for r in recs)/len(recs)),"mean tokens %.0f"%(sum(r["usage"]["completion_tokens"] for r in recs)/len(recs)))
with open(f"{S}/student_basephi4.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
json.dump(json.load(open(f"{S}/student_basephi4.shard0.jsonl.manifest.json")),open(f"{S}/student_basephi4.jsonl.manifest.json","w"),indent=1)
PY
for arm in translated random mined; do
  contrastforge prepare --inputs $D/train.jsonl $D/validation.jsonl --rollouts $S/data/student_basephi4.jsonl --strategy $arm --targets worked \
    --tokenizer $M --token-budget 3000000 --out $S/data/curriculaphi4/$arm 2>&1 | grep -E '"training_rows"|unique_training|Error' | tr -d '\n'; echo " <- $arm"
done
python - <<'PY'
import json
base=json.load(open("configs/contrast_sft_mn5.json"))
for arm in ("translated","random","mined"):
    c=dict(base); c["model"]="/gpfs/scratch/bsc21/bsc021885/cf/models/phi-4"
    c["target_modules"]=["qkv_proj","o_proj","gate_up_proj","down_proj"]   # Phi3 architecture (fused projections)
    c["train_data"]=f"/gpfs/scratch/bsc21/bsc021885/cf/data/curriculaphi4/{arm}/sft_train.jsonl"; c["validation_data"]=f"/gpfs/scratch/bsc21/bsc021885/cf/data/curriculaphi4/{arm}/sft_validation.jsonl"
    c["output_dir"]=f"/gpfs/scratch/bsc21/bsc021885/cf/runs/sftphi4_{arm}"; json.dump(c,open(f"configs/vphi4_{arm}.json","w"),indent=1)
print("phi4 configs written")
PY
cat > $L/sftphi4.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sftphi4
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
  R=""; [ -d \$S/runs/sftphi4_\$arm ] && ls \$S/runs/sftphi4_\$arm | grep -q checkpoint- && R="--resume latest"
  [ -f \$S/runs/sftphi4_\$arm/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftphi4-\$arm.out --error=$L/sftphi4-\$arm.err contrastforge train --config configs/vphi4_\$arm.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
for c in 1 2 3; do
  done_all=1; for arm in translated random mined; do [ -f $S/runs/sftphi4_$arm/metrics.json ] || done_all=0; done
  [ $done_all = 1 ] && break
  J=$(submit $L/sftphi4.sbatch); echo "== sftphi4 round$c $J $(date)"; waitjob $J
  for arm in translated random mined; do echo "$arm: $(cat $S/runs/sftphi4_$arm/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done
  tail -3 $L/sftphi4-random.err
done
# --- eval on test_eval12 (1,248 prompts, 4 shards): job A = base + random, job B = mined + translated
mkeval() { cat > $L/evalphi4_$1.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-evalphi4-$1
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
gpu() { for n in \$2; do A=""; [ \$n != base ] && A="--adapter \$S/runs/sftphi4_\$n"; [ -f \$E/test_\$n.shard\$3.jsonl.manifest.json ] && continue
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
    J=$(submit $L/evalphi4_$half.sbatch); echo "== evalphi4 $half round$c $J $(date)"; waitjob $J
  done
done
T=$D/test_eval12.jsonl
for n in base random mined translated; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok ($(wc -l < $E/test_$n.jsonl) rows)"; done
contrastforge compare --reports $E/report12_base.json $E/report12_translated.json $E/report12_random.json $E/report12_mined.json --out $E/comparison12.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalphi4"
for n in ("base","translated","random","mined"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"PHI4 {n:11s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} base_view={b['per_view']['base']:.2f} cf={b['per_view']['counterfactual']:.2f} tok={b['mean_completion_tokens']:.0f}"); print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())})
for x in json.load(open(f"{E}/comparison12.json"))["comparisons"]: print(x["treatment"].split("/")[-1],round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
echo "CHAIN DONE $(date)"
