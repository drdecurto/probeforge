S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200
for i in $(seq 1 60); do squeue -h -j 47135605 2>/dev/null | grep -q . || break; sleep 30; done
date; sacct -j 47135605 -X -o jobid,state,elapsed -n
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
python - <<'PY'
import json
S="/gpfs/scratch/bsc21/bsc021885/cf/data"; D=S+"/contrast13m_200"
want=set(json.load(open(f"{D}/mining_ids.json")))
recs=[json.loads(l) for k in range(4) for l in open(f"{S}/student_base_v2.shard{k}.jsonl")]
have={r["id"] for r in recs}; fps={r["run_fingerprint"] for r in recs}
print("records",len(recs),"ids collected",len(have&want),"of",len(want),"fingerprints",len(fps))
with open(f"{S}/student_base_v2.jsonl","w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
m=json.load(open(f"{S}/student_base_v2.shard0.jsonl.manifest.json")); json.dump(m,open(f"{S}/student_base_v2.jsonl.manifest.json","w"),indent=1)
ok=sum(r["verification"]["correct"] for r in recs)/len(recs); print("base accuracy on v2 mining set: %.3f"%ok)
rows={}
for p in ("train","validation"):
    for l in open(f"{D}/{p}.jsonl"): r=json.loads(l); rows[r["id"]]=r
import collections
c=collections.defaultdict(list)
for r in recs: c[rows[r["id"]]["language"]].append(r["verification"]["correct"])
print("per language:",{k:round(sum(v)/len(v),3) for k,v in sorted(c.items())})
c=collections.defaultdict(list)
for r in recs: c[rows[r["id"]]["view"]].append(r["verification"]["correct"])
print("per view:",{k:round(sum(v)/len(v),3) for k,v in c.items()})
PY
for arm in translated random mined no_counterfactual; do
  contrastforge prepare --inputs $D/train.jsonl $D/validation.jsonl --rollouts $S/data/student_base_v2.jsonl \
    --strategy $arm --targets worked --tokenizer $S/models/Qwen3-8B --token-budget 3000000 --out $S/data/curriculav2/$arm 2>&1 | grep -E '"training_rows"|"unique_training|sequence_tokens|Error|error' | tr -d '\n'; echo " <- $arm"
done
JOB=$(sbatch --parsable $S/logs/sftv2_4arms.sbatch); echo "submitted sftv2 $JOB"
squeue -u $USER -o '%.14i %.9P %.16j %.12q %.2t %.10M %R'
