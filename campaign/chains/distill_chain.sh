S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; D=$S/data/contrast13m_200; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13
M=$S/models/Qwen3-14B; E=$S/eval14b; ET=$S/evalthink; ET6=$S/evalthink60
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; [ -f "$1" ] || { echo "MISSING sbatch file $1" >&2; return 1; }
  for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
echo "START $(date)"
# ---------- A) 1,600 training-row prompts for base14b thinking rollouts (balanced by language)
python - <<'PY'
import json,random,collections
S="/gpfs/scratch/bsc21/bsc021885/cf"; D=f"{S}/data/contrast13m_200"
rows=[json.loads(l) for l in open(f"{S}/data/curricula14b/random/sft_train.jsonl")]
by=collections.defaultdict(list)
for r in rows: by[r["language"]].append(r["id"])
rng=random.Random(7); ids=[]
for lang,v in sorted(by.items()): rng.shuffle(v); ids+=v[:1600//13+1]
ids=ids[:1600]; rng.shuffle(ids); print("think-collect ids",len(ids),"langs",len(by))
json.dump(ids,open(f"{D}/tshard_all.json","w"))
for k in range(4): json.dump(ids[k::4],open(f"{D}/tshard{k}.json","w"))
PY
sed -e 's/cf-collect14b/cf-collect14b-think/' -e 's#mshard#tshard#g' -e 's#student_base14b\.shard#student_base14b_think.shard#g' -e 's#collect-14b-shard#collect-14bthink-shard#g' -e 's#--budgets 512 --samples 1#--budgets 4096 --samples 1 --thinking#' $L/collect_14b_debug.sbatch > $L/collect_14b_think.sbatch
grep -E "job-name|thinking" $L/collect_14b_think.sbatch | head -2
complete() { python3 - <<'PY'
import json,os
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D=S+"/contrast13m_200"; ok=True
for k in range(4):
    n=len(json.load(open(f"{D}/tshard{k}.json"))); f=f"{S}/student_base14b_think.shard{k}.jsonl"
    h=sum(1 for _ in open(f)) if os.path.exists(f) else 0; print(f"shard{k} {h}/{n}"); ok=ok and h>=n
print("COMPLETE" if ok else "INCOMPLETE")
PY
}
for c in 1 2 3 4 5; do complete | grep -q INCOMPLETE || break; J=$(submit $L/collect_14b_think.sbatch); echo "== think collect chunk$c $J $(date)"; waitjob $J; complete | tail -1; done
complete | tail -5
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"
recs=[json.loads(l) for k in range(4) for l in open(f"{S}/student_base14b_think.shard{k}.jsonl")]
ok=[r for r in recs if r["verification"]["correct"] and r["finish_reason"]=="stop" and r.get("thinking")]
tt=[r["usage"]["thinking_tokens"] or 0 for r in ok]
print(f"think rollouts {len(recs)} correct+finished {len(ok)} ({len(ok)/max(1,len(recs)):.2f}) mean think tokens of kept {sum(tt)/max(1,len(tt)):.0f} max {max(tt) if tt else 0}")
with open(f"{S}/student_base14b_think.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
PY
# ---------- B) curricula: distill (distilled traces only) and distillsyn (distilled + synthetic fill to 50 %)
python scripts/make_mixed.py $S/data/curricula14b/random $D $S/data/curricula14b/distill --frac 0.5 --seed 1 --traces $S/data/student_base14b_think.jsonl --fill none --max-trace-chars 5000
python scripts/make_mixed.py $S/data/curricula14b/random $D $S/data/curricula14b/distillsyn --frac 0.5 --seed 1 --traces $S/data/student_base14b_think.jsonl --fill synthetic --max-trace-chars 5000
python - <<'PY'
import json
from transformers import AutoTokenizer
from probeforge.training_data import encode_row
S="/gpfs/scratch/bsc21/bsc021885/cf"; tok=AutoTokenizer.from_pretrained(f"{S}/models/Qwen3-14B")
for arm in ("distill","distillsyn"):
    rows=[json.loads(l) for l in open(f"{S}/data/curricula14b/{arm}/sft_train.jsonl")]
    lens=[len(encode_row(tok,r,8192)["input_ids"]) for r in rows]; print(arm,"rows",len(rows),"max len",max(lens),"mean %.0f"%(sum(lens)/len(lens)),"over4096",sum(l>4096 for l in lens))
    c=json.load(open("configs/v14b_random.json"))
    c.update(train_data=f"{S}/data/curricula14b/{arm}/sft_train.jsonl",validation_data=f"{S}/data/curricula14b/{arm}/sft_validation.jsonl",output_dir=f"{S}/runs/sft14b_{arm}",max_length=4096)
    json.dump(c,open(f"configs/v14b_{arm}.json","w"),indent=1)
print("configs written (max_length 4096)")
PY
sed -e 's/cf-sft14b-mixed/cf-sft14b-distill/' -e 's/for n in mixed50 think100/for n in distill distillsyn/' $L/sft14b_mixed.sbatch > $L/sft14b_distill.sbatch
sed -e 's/cf-eval14b-mixed/cf-eval14b-distill/' -e 's/for n in mixed50 think100/for n in distill distillsyn/' -e 's/th 0 mixed50/th 0 distill/' -e 's/th 1 think100/th 1 distillsyn/' $L/eval14b_mixed.sbatch > $L/eval14b_distill.sbatch
grep -c "distill" $L/sft14b_distill.sbatch $L/eval14b_distill.sbatch
for c in 1 2 3; do ok=1; for n in distill distillsyn; do [ -f $S/runs/sft14b_$n/metrics.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/sft14b_distill.sbatch); echo "== sft14b distill round$c $J $(date)"; waitjob $J
  for n in distill distillsyn; do echo "$n: $(cat $S/runs/sft14b_$n/metrics.json 2>/dev/null | tr -d '\n ' | grep -o '"eval_loss":[0-9.]*')"; done; tail -2 $L/sft14b-distill.err; done
for c in 1 2; do ok=1; for n in distill distillsyn; do for sh in 0 1 2 3; do [ -f $E/test_$n.shard$sh.jsonl.manifest.json ] || ok=0; done; [ -f $ET/test_$n.jsonl.manifest.json ] || ok=0; [ -f $ET6/test_$n.jsonl.manifest.json ] || ok=0; done; [ $ok = 1 ] && break
  J=$(submit $L/eval14b_distill.sbatch); echo "== eval14b distill round$c $J $(date)"; waitjob $J; done
T=$D/test_eval12.jsonl
for n in distill distillsyn; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok"; done
contrastforge compare --reports $E/report12_random.json $E/report12_distill.json $E/report12_distillsyn.json --out $E/comparison12_distill.json >/dev/null
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf"; E=f"{S}/eval14b"
for n in ("random","mixed50","distill","distillsyn"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"DISTILL no-think {n:10s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} tok={b['mean_completion_tokens']:.0f}")
for x in json.load(open(f"{E}/comparison12_distill.json"))["comparisons"]: print(x["treatment"].split("/")[-1],"vs random:",round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
for label,Edir,names in (("think26",f"{S}/evalthink",("base14b","random14b","mixed50","distill","distillsyn")),("think60",f"{S}/evalthink60",("base14b","random14b","mixed50","distill","distillsyn"))):
    for n in names:
        try: recs=[json.loads(l) for l in open(f"{Edir}/test_{n}.jsonl")]
        except Exception: print(label,n,"missing"); continue
        ok=sum(r["verification"]["correct"] for r in recs); fin=sum(r["finish_reason"]=="stop" for r in recs)
        tt=[r["usage"].get("thinking_tokens") or 0 for r in recs]; sec=sum(r["generation_request_seconds"] for r in recs)/len(recs)
        print(f"DISTILL {label} {n:11s} n={len(recs)} correct={ok} ({ok/len(recs):.2f}) finished={fin} mean_think_tokens={sum(tt)/len(tt):.0f} mean_s={sec:.0f}")
PY
echo "CHAIN DONE $(date)"
