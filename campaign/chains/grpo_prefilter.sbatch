#!/usr/bin/env bash
#SBATCH --job-name=grpo-pre
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; G=$S/grpo; unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMEXPR_MAX_THREADS=64 VLLM_WORKER_MULTIPROC_METHOD=spawn
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k $S/envs/vllm32/bin/python $G/prefilter.py --model $S/models/merged/v4c-8b_s1 --shard $k --nshards 4 --pool 16000 --k 4 --out $G/pool8b > $G/pre_$k.log 2>&1 & done; wait
cat $G/pool8b/pool_*.jsonl > $G/pool8b/all.jsonl
python3 - <<'PY'
import json,collections
rows=[json.loads(l) for l in open("/gpfs/scratch/bsc21/bsc021885/cf/grpo/pool8b/all.jsonl")]
c=collections.Counter(r["p_correct"] for r in rows); print("p_correct histogram", sorted(c.items()))
hard=[r for r in rows if 0 < r["p_correct"] < 1]; zero=[r for r in rows if r["p_correct"]==0]
print("mixed", len(hard), "all-wrong", len(zero), "views(mixed)", collections.Counter(r["view"] for r in hard).most_common(), "families", collections.Counter(r["family"] for r in hard).most_common())
with open("/gpfs/scratch/bsc21/bsc021885/cf/grpo/pool8b/mixed.jsonl","w") as f:
    for r in hard: f.write(json.dumps(r,ensure_ascii=False)+"\n")
PY
