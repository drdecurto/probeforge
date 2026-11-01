S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; P=$S/data/packs; C=$P/probeforge_eval_candidate_2026-10-13; E=$S/evalpacks/core60
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
need() { for sh in $2; do f=$E/test_$1.shard$sh.jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 780 ] || return 0; done; return 1; }
echo "START $(date)"
for tag in base14b random14b; do
  for half in A B; do
    [ $half = A ] && shards="0 1 2 3" || shards="4 5 6 7"
    # reuse a job already queued/running with this name
    J=$(squeue -h -u $USER -n c60-$tag-$half -o %i | head -1)
    if need $tag "$shards"; then
      [ -n "$J" ] && echo "== c60-$tag-$half reusing queued job $J" || { J=$(submit $L/c60-$tag-$half.sbatch); echo "== c60-$tag-$half submitted $J $(date)"; }
      waitjob $J
    else echo "== c60-$tag-$half already complete"; fi
  done
  cat $E/test_$tag.shard*.jsonl > $E/test_$tag.jsonl; cp $E/test_$tag.shard0.jsonl.manifest.json $E/test_$tag.jsonl.manifest.json
  contrastforge evaluate --inputs $C/test_core60.jsonl --rollouts $E/test_$tag.jsonl --out $E/report_$tag.json >/dev/null && echo "report_$tag ok ($(wc -l < $E/test_$tag.jsonl) rows)"
done
contrastforge compare --reports $E/report_base8b.json $E/report_mined8b.json --out $E/comparison_8b.json >/dev/null
contrastforge compare --reports $E/report_base14b.json $E/report_random14b.json --out $E/comparison_14b.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalpacks/core60"
for n in ("base8b","mined8b","base14b","random14b"):
    b=json.load(open(f"{E}/report_{n}.json"))["budgets"]["512"]
    print(f"CORE60 {n:10s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} all_views={b['all_views_correct']:.3f} ci={[round(x,3) for x in b['group_mean_accuracy_ci95']]} tok={b['mean_completion_tokens']:.0f}")
    print("   view",{k:round(v,2) for k,v in b["per_view"].items()}); print("   family",{k:round(v,2) for k,v in b["per_family"].items()}); print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())})
for f in ("comparison_8b","comparison_14b"):
    for x in json.load(open(f"{E}/{f}.json"))["comparisons"]: print(f, round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]],"groups",x["groups"])
PY
echo "CHAIN DONE $(date)"
