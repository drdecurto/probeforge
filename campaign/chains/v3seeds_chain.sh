S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; D=$S/data/contrast13m_200; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13
M=$S/models/Qwen3-14B; E=$S/eval14b; ET=$S/evalthink60; mkdir -p $ET
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; [ -f "$1" ] || { echo "MISSING sbatch file $1" >&2; return 1; }
  for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
echo "START $(date)"
# ---------- configs + sbatch files up front
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf"
for arm in ("random","mined"):
    for seed in (2,3):
        c=json.load(open(f"configs/v3_14b_{arm}.json")); c.update(seed=seed,output_dir=f"{S}/runs/sftv3_14b_{arm}_s{seed}")
        json.dump(c,open(f"configs/v3_14b_{arm}_s{seed}.json","w"),indent=1)
print("seed configs written")
PY
cat > $L/sftv3_14b_seeds.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv3-14b-seeds
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
source ~/probeforge/slurm/environment.sh; S=$S; cd ~/probeforge; K=0
for n in random_s2 random_s3 mined_s2 mined_s3; do
  R=""; [ -d \$S/runs/sftv3_14b_\$n ] && ls \$S/runs/sftv3_14b_\$n | grep -q checkpoint- && R="--resume latest"
  [ -f \$S/runs/sftv3_14b_\$n/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftv3_14b-\$n.out --error=$L/sftv3_14b-\$n.err contrastforge train --config configs/v3_14b_\$n.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
mkeval() { cat > $L/$1.sbatch <<SB
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
source ~/probeforge/slurm/environment.sh; S=$S; D=$D; T=\$D/test_eval12.jsonl; E=$E; M=$M; cd ~/probeforge
gpu() { for n in \$2; do [ -f \$E/test_v3_\$n.shard\$3.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_v3\${n}_s\$3.out --error=\$E/collect_v3\${n}_s\$3.err \\
  contrastforge collect --inputs \$T --only-ids \$D/eshard\$3.json --out \$E/test_v3_\$n.shard\$3.jsonl --model \$M --adapter \$S/runs/sftv3_14b_\$n --budgets 512 --samples 1; done; }
gpu 0 "$2" 0 & gpu 1 "$2" 1 & gpu 2 "$2" 2 & gpu 3 "$2" 3 &
wait; echo "end \$(date)"
SB
}
mkeval cf-evalv3-14b-sA "random_s2 mined_s2"; mkeval cf-evalv3-14b-sB "random_s3 mined_s3"
# thinking eval on diag60: 2 base-view + 2 counterfactual-view prompts per language = 52
python - <<'PY'
import json,collections
C="/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_eval_candidate_2026-10-13"
rows=[json.loads(l) for l in open(f"{C}/test_diagnostic60.jsonl")]; c=collections.Counter(); ids=[]
for r in rows:
    if r["view"] in ("base","counterfactual") and c[(r["language"],r["view"])]<2: c[(r["language"],r["view"])]+=1; ids.append(r["id"])
json.dump(ids,open(f"{C}/think60_ids.json","w")); print("think60 ids",len(ids))
PY
cat > $L/think60.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-think60
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
source ~/probeforge/slurm/environment.sh; S=$S; C=$C; T=\$C/test_diagnostic60.jsonl; E=$ET; cd ~/probeforge
run() { [ -f \$E/test_\$2.jsonl.manifest.json ] && return; CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_\$2.out --error=\$E/collect_\$2.err \\
  contrastforge collect --inputs \$T --only-ids \$C/think60_ids.json --out \$E/test_\$2.jsonl --model \$S/models/\$3 \$4 --budgets 4096 --samples 1 --thinking; }
run 0 mined8b Qwen3-8B "--adapter \$S/runs/sftv2_mined_s1" & run 1 base14b Qwen3-14B "" & run 2 random14b Qwen3-14B "--adapter \$S/runs/sft14b_random" & run 3 v3mined14b Qwen3-14B "--adapter \$S/runs/sftv3_14b_mined" &
wait; echo "end \$(date)"
SB
ls $L/sftv3_14b_seeds.sbatch $L/cf-evalv3-14b-s?.sbatch $L/think60.sbatch
# ---------- A) SFT seeds (4 adapters, resumable)
for c in 1 2 3; do
  ok=1; for n in random_s2 random_s3 mined_s2 mined_s3; do [ -f $S/runs/sftv3_14b_$n/metrics.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/sftv3_14b_seeds.sbatch); echo "== sftv3 14B seeds round$c $J $(date)"; waitjob $J
  for n in random_s2 random_s3 mined_s2 mined_s3; do echo "$n: $(cat $S/runs/sftv3_14b_$n/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done
done
# ---------- B) eval seeds on test_eval12
for half in A B; do
  [ $half = A ] && names="random_s2 mined_s2" || names="random_s3 mined_s3"
  for c in 1 2; do
    ok=1; for n in $names; do for sh in 0 1 2 3; do [ -f $E/test_v3_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; done; [ $ok = 1 ] && break
    J=$(submit $L/cf-evalv3-14b-s$half.sbatch); echo "== evalv3 seeds $half round$c $J $(date)"; waitjob $J
  done
done
T=$D/test_eval12.jsonl
for n in random_s2 random_s3 mined_s2 mined_s3; do cat $E/test_v3_$n.shard*.jsonl > $E/test_v3_$n.jsonl; cp $E/test_v3_$n.shard0.jsonl.manifest.json $E/test_v3_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_v3_$n.jsonl --out $E/report12_v3_$n.json >/dev/null && echo "report12_v3_$n ok ($(wc -l < $E/test_v3_$n.jsonl) rows)"; done
python - <<'PY'
import json,statistics as st
E="/gpfs/scratch/bsc21/bsc021885/cf/eval14b"
def rep(n): return json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
for label,names in (("v2 random",("random","random_s1","random_s2")),("v2 mined",("mined","mined_s1","mined_s2")),("v3 random",("v3_random","v3_random_s2","v3_random_s3")),("v3 mined",("v3_mined","v3_mined_s2","v3_mined_s3"))):
    v=[rep(n)["pass_at_1"] for n in names]; w=[rep(n)["worst_language"] for n in names]; b=[rep(n)["per_view"]["boundary"] for n in names]
    print(f"SEEDS14B {label:10s} pass@1 {[round(x,3) for x in v]} mean {st.mean(v):.3f} sd {st.stdev(v):.3f} | worst mean {st.mean(w):.3f} | boundary mean {st.mean(b):.2f}")
PY
# ---------- C) thinking-mode on diag60 (52 prompts)
for c in 1 2; do ok=1; for n in mined8b base14b random14b v3mined14b; do [ -f $ET/test_$n.jsonl.manifest.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/think60.sbatch); echo "== think60 round$c $J $(date)"; waitjob $J; done
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalthink60"; C="/gpfs/scratch/bsc21/bsc021885/cf/data/packs/probeforge_eval_candidate_2026-10-13"
view={json.loads(l)["id"]:json.loads(l)["view"] for l in open(f"{C}/test_diagnostic60.jsonl")}
for n in ("mined8b","base14b","random14b","v3mined14b"):
    try: recs=[json.loads(l) for l in open(f"{E}/test_{n}.jsonl")]
    except Exception as e: print(n,"missing"); continue
    ok=sum(r["verification"]["correct"] for r in recs); fin=sum(r["finish_reason"]=="stop" for r in recs)
    tt=[r["usage"].get("thinking_tokens") or 0 for r in recs]; sec=sum(r["generation_request_seconds"] for r in recs)/len(recs)
    pv={}
    for v in ("base","counterfactual"):
        rr=[r for r in recs if view[r["id"]]==v]; pv[v]=round(sum(r["verification"]["correct"] for r in rr)/max(1,len(rr)),2)
    print(f"THINK60 {n:11s} n={len(recs)} correct={ok} ({ok/len(recs):.2f}) base={pv['base']} cf={pv['counterfactual']} finished={fin} mean_think_tokens={sum(tt)/len(tt):.0f} mean_s={sec:.0f}")
PY
echo "CHAIN DONE $(date)"
