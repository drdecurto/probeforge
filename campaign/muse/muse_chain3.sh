S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; E=$S/evalmuse
D=$S/data/contrast13m_200; P=$S/data/packs; CP=$P/probeforge_eval_candidate_2026-10-13; V=$P/probeforge_coverage_extension_2026-10-13
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>$L/sbatch.err) && break; [ $i = 1 ] && cat $L/sbatch.err >&2; sleep 120; done; echo "$J"; }
remaining_all() { local t=0; for spec in "think26 $D/test_eval12.jsonl" "think60 $CP/test_diagnostic60.jsonl" "main4k $D/test_eval12.jsonl" "dev112 $V/development112_native.jsonl" "diag60 $CP/test_diagnostic60.jsonl" "core60 $CP/test_core60.jsonl"; do
  set -- $spec; r=$(python $E/remaining.py $1 $2 1); echo "  $1 remaining $r" >&2; t=$((t+r)); done; rm -f $E/*_r0.json; echo $t; }
echo "START $(date)"

for c in 1 2 3 4; do
  t=$(remaining_all); echo "total remaining $t $(date)"; [ "$t" = 0 ] && break
  J=$(submit $L/muse_serve3.sbatch); echo "== serve2 round$c $J $(date)"; waitjob $J
  grep -E "^nodes|server up|server not|set .* remaining|all sets" $L/muse-serve-$J.out | head -12
done
for n in think26 think60 main4k dev112 diag60 core60; do
  ls $E/test_$n.*jsonl $E/test_$n.jsonl 2>/dev/null | grep -v manifest | xargs cat 2>/dev/null | python -c "
import sys,json
seen=set(); out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['id'] in seen: continue
    seen.add(r['id']); out.append(l)
sys.stdout.write(''.join(out))" > $E/all_$n.jsonl
  m=$(ls $E/test_$n.*jsonl.manifest.json $E/test_$n.jsonl.manifest.json 2>/dev/null | head -1); [ -n "$m" ] && cp $m $E/all_$n.jsonl.manifest.json
  echo "$n merged $(wc -l < $E/all_$n.jsonl)"
done
contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $E/all_main4k.jsonl --out $E/report12_muse.json >/dev/null && echo "report12_muse ok"
contrastforge evaluate --inputs $CP/test_core60.jsonl --rollouts $E/all_core60.jsonl --out $E/report_core60_muse.json >/dev/null && echo "core60 ok"
contrastforge evaluate --inputs $CP/test_diagnostic60.jsonl --rollouts $E/all_diag60.jsonl --out $E/report_diag60_muse.json >/dev/null && echo "diag60 ok"
python scripts/rescore_lenient.py $V/development112_native.jsonl $E/all_dev112.jsonl --out $E/lenient_dev112_muse.json | grep -E "^all|^status"
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalmuse"
for f,label in (("report12_muse","MAIN"),("report_core60_muse","CORE60"),("report_diag60_muse","DIAG60")):
    try: b=list(json.load(open(f"{E}/{f}.json"))["budgets"].values())[0]
    except Exception as e: print(label,"missing",e); continue
    print(f"MUSE {label:6s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} ci={[round(x,3) for x in b['group_mean_accuracy_ci95']]} tok={b['mean_completion_tokens']:.0f} length_stops={b['length_stops']} missing={b['missing_requests']}")
    print("   view",{k:round(v,2) for k,v in b["per_view"].items()}); print("   family",{k:round(v,2) for k,v in b["per_family"].items()}); print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())})
for n in ("think26","think60"):
    recs=[json.loads(l) for l in open(f"{E}/all_{n}.jsonl")]; ok=sum(r["verification"]["correct"] for r in recs)
    ct=sum(((r.get("usage") or {}).get("completion_tokens") or 0) for r in recs)/max(1,len(recs))
    print(f"MUSE {n} n={len(recs)} correct={ok} ({ok/max(1,len(recs)):.2f}) finished={sum(r['finish_reason']=='stop' for r in recs)} mean_completion_tokens={ct:.0f}")
PY
echo "CHAIN DONE $(date)"
