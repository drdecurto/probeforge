S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; L=$S/logs; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J="" err=""; [ -f "$1" ] || { echo "MISSING sbatch file $1" >&2; return 1; }
  for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
echo "START $(date)"
# ================= A) on-policy rollouts: sftv2_mined_s1 on the 7,748 mining prompts
sed -e 's/cf-collect-v2d/cf-collect-onp/' -e 's#student_base_v2\.shard#student_onp.shard#' -e 's#collect-v2d-shard#collect-onp-shard#' \
    -e 's#--budgets 512#--adapter $S/runs/sftv2_mined_s1 --budgets 512#' $L/collect_v2_debug.sbatch > $L/collect_onp_debug.sbatch
grep -E "job-name|adapter" $L/collect_onp_debug.sbatch | head -2
complete() { python3 - <<'PY'
import json,os
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D=S+"/contrast13m_200"; ok=True
for k in range(4):
    n=len(json.load(open(f"{D}/mshard{k}.json"))); f=f"{S}/student_onp.shard{k}.jsonl"
    h=sum(1 for _ in open(f)) if os.path.exists(f) else 0; print(f"shard{k} {h}/{n}"); ok=ok and h>=n
print("COMPLETE" if ok else "INCOMPLETE")
PY
}
for c in 1 2 3; do
  complete | grep -q INCOMPLETE || break
  J=$(submit $L/collect_onp_debug.sbatch); echo "== collect onp chunk$c $J $(date)"; waitjob $J; complete | tail -1
done
complete | grep -q COMPLETE || { echo "on-policy collect incomplete, skipping DPO"; SKIPDPO=1; }
if [ -z "${SKIPDPO:-}" ]; then
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"
recs=[json.loads(l) for k in range(4) for l in open(f"{S}/student_onp.shard{k}.jsonl")]
print("onp records",len(recs),"fingerprints",len({r["run_fingerprint"] for r in recs}),"SFT-student acc on mining set %.3f"%(sum(r["verification"]["correct"] for r in recs)/len(recs)))
with open(f"{S}/student_onp.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
json.dump(json.load(open(f"{S}/student_onp.shard0.jsonl.manifest.json")),open(f"{S}/student_onp.jsonl.manifest.json","w"),indent=1)
PY
# ================= B) preferences from the SFT model's own verified errors
contrastforge prepare --inputs $D/train.jsonl $D/validation.jsonl --rollouts $S/data/student_onp.jsonl --strategy mined --targets worked \
  --tokenizer $S/models/Qwen3-8B --token-budget 3000000 --out $S/data/curriculav2/onp_dpo 2>&1 | grep -E 'preference_counts' -A3 | tr -d '\n '; echo
wc -l $S/data/curriculav2/onp_dpo/preferences_train.jsonl $S/data/curriculav2/onp_dpo/preferences_validation.jsonl
head -1 $S/data/curriculav2/onp_dpo/preferences_train.jsonl | python3 -c "import sys,json; r=json.loads(sys.stdin.readline()); print('chosen:',r['chosen'][:100]); print('rejected:',r['rejected'][:140]); print('reason:',r['preference_reason'])"
python - <<'PY'
import json
c=json.load(open("configs/v2_dpo_mined.json")); S="/gpfs/scratch/bsc21/bsc021885/cf"
c.update(train_data=f"{S}/data/curriculav2/onp_dpo/reference_train.jsonl",validation_data=f"{S}/data/curriculav2/onp_dpo/reference_validation.jsonl",
         output_dir=f"{S}/runs/dpo_onp",init_adapter=f"{S}/runs/sftv2_mined_s1",learning_rate=2e-6)
json.dump(c,open("configs/v2_dpo_onp.json","w"),indent=1); print("dpo config:",{k:c[k] for k in ("learning_rate","beta","epochs","max_length") if k in c})
PY
sed -e 's/cf-dpo-v2/cf-dpo-onp/' -e 's#curriculav2/mined_dpo#curriculav2/onp_dpo#g' -e 's#v2_dpo_mined.json#v2_dpo_onp.json#g' -e 's#runs/dpov2_mined#runs/dpo_onp#g' $L/dpo_v2.sbatch > $L/dpo_onp.sbatch
grep -cE "onp" $L/dpo_onp.sbatch
for c in 1 2; do [ -f $S/runs/dpo_onp/metrics.json ] && break; J=$(submit $L/dpo_onp.sbatch); echo "== dpo onp round$c $J $(date)"; waitjob $J; done
cat $S/runs/dpo_onp/metrics.json 2>/dev/null | tr -d '\n ' | cut -c1-300; echo
sed -e 's/cf-eval-dpo/cf-eval-dpo-onp/' -e 's#runs/dpov2_mined#runs/dpo_onp#g' -e 's#test_dpo_mined#test_dpo_onp#g' -e 's#collect_dpo_s#collect_dpoonp_s#g' -e 's#report12_dpo_mined#report12_dpo_onp#g' -e 's#comparison12_dpo\.json#comparison12_dpo_onp.json#g' -e 's#"dpo_mined"#"dpo_onp"#' $L/eval_dpo.sbatch > $L/eval_dpo_onp.sbatch
J=$(submit $L/eval_dpo_onp.sbatch); echo "== eval dpo onp $J $(date)"; waitjob $J; grep -E "^DPO|dpo vs sft" $L/cf-eval-dpo-onp-$J.out
fi
# ================= C) diag60 on the final adapters (sbatch files generated up front)
E=$S/evalpacks/diag60; mkdir -p $E
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
source ~/probeforge/slurm/environment.sh; S=$S; P=$P; C=$C; E=$E; cd ~/probeforge
A=""; [ -n "$4" ] && A="--adapter $4"; K=0
for sh in $2; do CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_$5_s\$sh.out --error=\$E/collect_$5_s\$sh.err \\
  contrastforge collect --inputs \$C/test_diagnostic60.jsonl --only-ids \$P/diag60_shard\$sh.json --out \$E/test_$5.shard\$sh.jsonl --model $3 \$A --budgets 512 --samples 1 & K=\$((K+1)); done; wait
echo "end \$(date)"
SB
}
mk d60-mined8b-A "0 1 2 3" $S/models/Qwen3-8B "$S/runs/sftv2_mined_s1" mined8b; mk d60-mined8b-B "4 5 6 7" $S/models/Qwen3-8B "$S/runs/sftv2_mined_s1" mined8b
mk d60-random14b-A "0 1 2 3" $S/models/Qwen3-14B "$S/runs/sft14b_random" random14b; mk d60-random14b-B "4 5 6 7" $S/models/Qwen3-14B "$S/runs/sft14b_random" random14b
ls $L/d60-*.sbatch; ls $P/diag60_shard*.json | wc -l
need() { for sh in $2; do f=$E/test_$1.shard$sh.jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 780 ] || return 0; done; return 1; }
for tag in mined8b random14b; do
  for half in A B; do
    [ $half = A ] && shards="0 1 2 3" || shards="4 5 6 7"
    if need $tag "$shards"; then J=$(submit $L/d60-$tag-$half.sbatch); echo "== d60-$tag-$half $J $(date)"; waitjob $J; else echo "== d60-$tag-$half already complete"; fi
  done
  cat $E/test_$tag.shard*.jsonl > $E/test_$tag.jsonl; cp $E/test_$tag.shard0.jsonl.manifest.json $E/test_$tag.jsonl.manifest.json
  contrastforge evaluate --inputs $C/test_diagnostic60.jsonl --rollouts $E/test_$tag.jsonl --out $E/report_$tag.json >/dev/null && echo "report_$tag ok ($(wc -l < $E/test_$tag.jsonl) rows)"
done
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalpacks"
for n in ("mined8b","random14b"):
    for pack in ("core60","diag60"):
        try: b=json.load(open(f"{E}/{pack}/report_{n}.json"))["budgets"]["512"]
        except Exception as e: print(pack,n,"missing",e); continue
        print(f"{pack.upper()} {n:10s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} ci={[round(x,3) for x in b['group_mean_accuracy_ci95']]} tok={b['mean_completion_tokens']:.0f}")
        print("   view",{k:round(v,2) for k,v in b["per_view"].items()}); print("   family",{k:round(v,2) for k,v in b["per_family"].items()})
PY
echo "CHAIN DONE $(date)"
