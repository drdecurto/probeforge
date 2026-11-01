S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; D=$S/data/contrast13m_200; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13
M=$S/models/Qwen3-14B; E=$S/eval14b; ET=$S/evalthink; ET6=$S/evalthink60
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; [ -f "$1" ] || { echo "MISSING sbatch file $1" >&2; return 1; }
  for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
echo "START $(date)"
ADAPTERS="mixed50_s1 mixed50_s2 distill_s1 distill_s2 distillsyn_s1 distillsyn_s2"
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf"
for arm in ("mixed50","distill","distillsyn"):
    for seed in (1,2):
        c=json.load(open(f"configs/v14b_{arm}.json")); c.update(seed=seed,output_dir=f"{S}/runs/sft14b_{arm}_s{seed}")
        json.dump(c,open(f"configs/v14b_{arm}_s{seed}.json","w"),indent=1)
print("seed configs written (training seed only; curricula unchanged)")
PY
mksft() { cat > $L/$1.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=$1
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=$3
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:$3
#SBATCH --time=02:00:00
#SBATCH --output=$L/%x-%j.out
#SBATCH --error=$L/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=$S; cd ~/probeforge; K=0
for n in $2; do
  R=""; [ -d \$S/runs/sft14b_\$n ] && ls \$S/runs/sft14b_\$n | grep -q checkpoint- && R="--resume latest"
  [ -f \$S/runs/sft14b_\$n/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sft14b-\$n.out --error=$L/sft14b-\$n.err contrastforge train --config configs/v14b_\$n.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
}
mksft cf-sft14b-mixA "mixed50_s1 mixed50_s2 distill_s1 distill_s2" 4; mksft cf-sft14b-mixB "distillsyn_s1 distillsyn_s2" 2
# eval jobs: no-thinking shards (3 adapters per job, 3 runs per GPU ≈ 75 min) + short thinking runs for mixed50
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
nt() { for n in $2; do [ -f \$E/test_\$n.shard\$3.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_\${n}_s\$3.out --error=\$E/collect_\${n}_s\$3.err \\
  contrastforge collect --inputs \$T --only-ids \$D/eshard\$3.json --out \$E/test_\$n.shard\$3.jsonl --model \$M --adapter \$S/runs/sft14b_\$n --budgets 512 --samples 1; done; }
nt 0 "$2" 0 & nt 1 "$2" 1 & nt 2 "$2" 2 & nt 3 "$2" 3 &
wait; echo "end \$(date)"
SB
}
mkeval cf-eval14b-mixA "mixed50_s1 mixed50_s2 distill_s1"; mkeval cf-eval14b-mixB "distill_s2 distillsyn_s1 distillsyn_s2"
# thinking job: 4 long runs (distill_s1/s2, distillsyn_s1/s2) in parallel + the 2 short mixed50 runs chained
cat > $L/think_mixseeds.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-think-mixseeds
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
source ~/probeforge/slurm/environment.sh; S=$S; D=$D; T=\$D/test_eval12.jsonl; ET=$ET; ET6=$ET6; C=$C; M=$M; cd ~/probeforge
th() { n=\$2; [ -f \$ET/test_\$n.jsonl.manifest.json ] || CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$ET/collect_\$n.out --error=\$ET/collect_\$n.err \\
  contrastforge collect --inputs \$T --only-ids \$D/think_ids.json --out \$ET/test_\$n.jsonl --model \$M --adapter \$S/runs/sft14b_\$n --budgets 4096 --samples 1 --thinking
  [ -f \$ET6/test_\$n.jsonl.manifest.json ] || CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$ET6/collect_\$n.out --error=\$ET6/collect_\$n.err \\
  contrastforge collect --inputs \$C/test_diagnostic60.jsonl --only-ids \$C/think60_ids.json --out \$ET6/test_\$n.jsonl --model \$M --adapter \$S/runs/sft14b_\$n --budgets 4096 --samples 1 --thinking; }
(th 0 distill_s1; th 0 mixed50_s1) & (th 1 distill_s2; th 1 mixed50_s2) & th 2 distillsyn_s1 & th 3 distillsyn_s2 &
wait; echo "end \$(date)"
SB
ls $L/cf-sft14b-mix?.sbatch $L/cf-eval14b-mix?.sbatch $L/think_mixseeds.sbatch
# ---------- A) SFT
for job in cf-sft14b-mixA cf-sft14b-mixB; do
  [ $job = cf-sft14b-mixA ] && names="mixed50_s1 mixed50_s2 distill_s1 distill_s2" || names="distillsyn_s1 distillsyn_s2"
  for c in 1 2 3; do ok=1; for n in $names; do [ -f $S/runs/sft14b_$n/metrics.json ] || ok=0; done; [ $ok = 1 ] && break
    J=$(submit $L/$job.sbatch); echo "== $job round$c $J $(date)"; waitjob $J
    for n in $names; do echo "$n: $(cat $S/runs/sft14b_$n/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done; done
done
# ---------- B) no-thinking eval
for job in cf-eval14b-mixA cf-eval14b-mixB; do
  [ $job = cf-eval14b-mixA ] && names="mixed50_s1 mixed50_s2 distill_s1" || names="distill_s2 distillsyn_s1 distillsyn_s2"
  for c in 1 2; do ok=1; for n in $names; do for sh in 0 1 2 3; do [ -f $E/test_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; done; [ $ok = 1 ] && break
    J=$(submit $L/$job.sbatch); echo "== $job round$c $J $(date)"; waitjob $J; done
done
T=$D/test_eval12.jsonl
for n in $ADAPTERS; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok"; done
# ---------- C) thinking eval (2 rounds allowed: long distill traces may hit the 2-h limit; collect is resumable)
for c in 1 2 3; do ok=1; for n in $ADAPTERS; do [ -f $ET/test_$n.jsonl.manifest.json ] || ok=0; [ -f $ET6/test_$n.jsonl.manifest.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/think_mixseeds.sbatch); echo "== think mixseeds round$c $J $(date)"; waitjob $J; done
python - <<'PY'
import json,statistics as st
S="/gpfs/scratch/bsc21/bsc021885/cf"; E=f"{S}/eval14b"; C=f"{S}/data/packs/probeforge_eval_candidate_2026-10-13"
def rep(n): return json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
def think(d,n):
    try: recs=[json.loads(l) for l in open(f"{d}/test_{n}.jsonl")]
    except Exception: return None
    return sum(r["verification"]["correct"] for r in recs)/len(recs), sum((r["usage"].get("thinking_tokens") or 0) for r in recs)/len(recs)
groups={"random (v2)":["random","random_s1","random_s2"],"mixed50":["mixed50","mixed50_s1","mixed50_s2"],"distill":["distill","distill_s1","distill_s2"],"distillsyn":["distillsyn","distillsyn_s1","distillsyn_s2"]}
for g,names in groups.items():
    nt=[rep(n)["pass_at_1"] for n in names]; bd=[rep(n)["per_view"]["boundary"] for n in names]; wl=[rep(n)["worst_language"] for n in names]
    t26=[think(f"{S}/evalthink",n if n!="random" else "random14b") for n in names]; t60=[think(f"{S}/evalthink60",n if n!="random" else "random14b") for n in names]
    f=lambda v: f"{st.mean(v):.3f}±{st.stdev(v):.3f}" if len(v)>1 else f"{v[0]:.3f}"
    t26v=[x[0] for x in t26 if x]; t60v=[x[0] for x in t60 if x]; tk=[x[1] for x in t26 if x]
    print(f"MIXSEEDS {g:12s} no-think {f(nt)} (worst {st.mean(wl):.3f}, boundary {st.mean(bd):.2f}) | think26 {f(t26v) if t26v else 'n/a'} | think60 {f(t60v) if t60v else 'n/a'} | think tokens {st.mean(tk):.0f}" if tk else f"MIXSEEDS {g} no-think {f(nt)}")
    print("   per seed no-think", [round(x,3) for x in nt], "think26", [round(x,2) for x in t26v], "think60", [round(x,2) for x in t60v])
PY
echo "CHAIN DONE $(date)"
