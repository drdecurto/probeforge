S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; D=$S/data/contrast13m_200; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13
M=$S/models/Qwen3-14B; E=$S/eval14b; ET=$S/evalthink; ET6=$S/evalthink60
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; [ -f "$1" ] || { echo "MISSING sbatch file $1" >&2; return 1; }
  for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
echo "START $(date)"
# ---------- mixed-mode curricula from the 14B v2 random curriculum
python scripts/make_mixed.py $S/data/curricula14b/random $D $S/data/curricula14b/mixed50 --frac 0.5 --seed 1
python scripts/make_mixed.py $S/data/curricula14b/random $D $S/data/curricula14b/think100 --frac 1.0 --seed 1
python - <<'PY'
import json
from transformers import AutoTokenizer
from probeforge.training_data import encode_row
S="/gpfs/scratch/bsc21/bsc021885/cf"; tok=AutoTokenizer.from_pretrained(f"{S}/models/Qwen3-14B")
rows=[json.loads(l) for l in open(f"{S}/data/curricula14b/mixed50/sft_train.jsonl")]
t=next(r for r in rows if r.get("thinking")); n=next(r for r in rows if not r.get("thinking"))
for name,r in (("thinking",t),("plain",n)):
    e=encode_row(tok,r,2048); text=tok.decode(e["input_ids"]); i=text.find("<|im_start|>assistant"); print(name,"->",repr(text[i:i+90]), "len",len(e["input_ids"]))
lens=[len(encode_row(tok,r,4096)["input_ids"]) for r in rows]; print("max len",max(lens),"mean",sum(lens)/len(lens))
for arm in ("mixed50","think100"):
    c=json.load(open("configs/v14b_random.json"))
    c.update(train_data=f"{S}/data/curricula14b/{arm}/sft_train.jsonl",validation_data=f"{S}/data/curricula14b/{arm}/sft_validation.jsonl",output_dir=f"{S}/runs/sft14b_{arm}")
    json.dump(c,open(f"configs/v14b_{arm}.json","w"),indent=1)
print("configs written")
PY
cat > $L/sft14b_mixed.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-sft14b-mixed
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
for n in mixed50 think100; do
  R=""; [ -d \$S/runs/sft14b_\$n ] && ls \$S/runs/sft14b_\$n | grep -q checkpoint- && R="--resume latest"
  [ -f \$S/runs/sft14b_\$n/metrics.json ] || CUDA_VISIBLE_DEVICES=\$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sft14b-\$n.out --error=$L/sft14b-\$n.err contrastforge train --config configs/v14b_\$n.json \$R &
  K=\$((K+1))
done
wait; echo "end \$(date)"
SB
cat > $L/eval14b_mixed.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cf-eval14b-mixed
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
source ~/probeforge/slurm/environment.sh; S=$S; D=$D; T=\$D/test_eval12.jsonl; E=$E; ET=$ET; ET6=$ET6; C=$C; M=$M; cd ~/probeforge
nt() { for n in mixed50 think100; do [ -f \$E/test_\$n.shard\$2.jsonl.manifest.json ] && continue
  CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$E/collect_\${n}_s\$2.out --error=\$E/collect_\${n}_s\$2.err \\
  contrastforge collect --inputs \$T --only-ids \$D/eshard\$2.json --out \$E/test_\$n.shard\$2.jsonl --model \$M --adapter \$S/runs/sft14b_\$n --budgets 512 --samples 1; done; }
th() { n=\$2; [ -f \$ET/test_\$n.jsonl.manifest.json ] || CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$ET/collect_\$n.out --error=\$ET/collect_\$n.err \\
  contrastforge collect --inputs \$T --only-ids \$D/think_ids.json --out \$ET/test_\$n.jsonl --model \$M --adapter \$S/runs/sft14b_\$n --budgets 4096 --samples 1 --thinking
  [ -f \$ET6/test_\$n.jsonl.manifest.json ] || CUDA_VISIBLE_DEVICES=\$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=\$ET6/collect_\$n.out --error=\$ET6/collect_\$n.err \\
  contrastforge collect --inputs \$C/test_diagnostic60.jsonl --only-ids \$C/think60_ids.json --out \$ET6/test_\$n.jsonl --model \$M --adapter \$S/runs/sft14b_\$n --budgets 4096 --samples 1 --thinking; }
(nt 0 0; th 0 mixed50) & (nt 1 1; th 1 think100) & nt 2 2 & nt 3 3 &
wait; echo "end \$(date)"
SB
for c in 1 2 3; do ok=1; for n in mixed50 think100; do [ -f $S/runs/sft14b_$n/metrics.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/sft14b_mixed.sbatch); echo "== sft14b mixed round$c $J $(date)"; waitjob $J
  for n in mixed50 think100; do echo "$n: $(cat $S/runs/sft14b_$n/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done; tail -2 $L/sft14b-mixed50.err; done
for c in 1 2; do ok=1; for n in mixed50 think100; do for sh in 0 1 2 3; do [ -f $E/test_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; [ -f $ET/test_$n.jsonl.manifest.json ] || ok=0; [ -f $ET6/test_$n.jsonl.manifest.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/eval14b_mixed.sbatch); echo "== eval14b mixed round$c $J $(date)"; waitjob $J; done
T=$D/test_eval12.jsonl
for n in mixed50 think100; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok"; done
contrastforge compare --reports $E/report12_random.json $E/report12_mixed50.json $E/report12_think100.json --out $E/comparison12_mixed.json >/dev/null
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf"; E=f"{S}/eval14b"
for n in ("base14b","random","mixed50","think100"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"MIXED no-think {n:9s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} base_view={b['per_view']['base']:.2f} tok={b['mean_completion_tokens']:.0f}")
for x in json.load(open(f"{E}/comparison12_mixed.json"))["comparisons"]: print(x["treatment"].split("/")[-1],"vs random:",round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
C=f"{S}/data/packs/probeforge_eval_candidate_2026-10-13"
view={json.loads(l)["id"]:json.loads(l)["view"] for l in open(f"{C}/test_diagnostic60.jsonl")}
for label,Edir,names in (("think26",f"{S}/evalthink",("base14b","random14b","mixed50","think100")),("think60",f"{S}/evalthink60",("base14b","random14b","v3mined14b","mixed50","think100"))):
    for n in names:
        try: recs=[json.loads(l) for l in open(f"{Edir}/test_{n}.jsonl")]
        except Exception: print(label,n,"missing"); continue
        ok=sum(r["verification"]["correct"] for r in recs); fin=sum(r["finish_reason"]=="stop" for r in recs)
        tt=[r["usage"].get("thinking_tokens") or 0 for r in recs]; sec=sum(r["generation_request_seconds"] for r in recs)/len(recs)
        print(f"MIXED {label} {n:11s} n={len(recs)} correct={ok} ({ok/len(recs):.2f}) finished={fin} mean_think_tokens={sum(tt)/len(tt):.0f} mean_s={sec:.0f}")
PY
echo "CHAIN DONE $(date)"
