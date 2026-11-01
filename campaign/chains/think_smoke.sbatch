#!/usr/bin/env bash
#SBATCH --job-name=cf-think-smoke
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast60; cd ~/probeforge
mkdir -p $S/eval60think
srun --ntasks=1 --cpu-bind=none contrastforge collect --inputs $D/test.jsonl --only-ids $S/data/think_smoke_ids.json \
  --out $S/eval60think/base_think2048.jsonl --model $S/models/Qwen3-8B --budgets 2048 --samples 1 --thinking
contrastforge evaluate --inputs $D/test.jsonl --rollouts $S/eval60think/base_think2048.jsonl --out $S/eval60think/report_base_think2048.json > /dev/null
python - <<'PY'
import json
r=json.load(open("/gpfs/scratch/bsc21/bsc021885/cf/eval60think/report_base_think2048.json"))["budgets"]["2048"]
print("THINK pass@1",r["pass_at_1"],"per_view",r["per_view"],"per_lang",r["per_language"],"len_stops",r["length_stops"],"mean_tokens",r["mean_completion_tokens"])
recs=[json.loads(l) for l in open("/gpfs/scratch/bsc21/bsc021885/cf/eval60think/base_think2048.jsonl")]
tt=[x["usage"]["thinking_tokens"] for x in recs if x["usage"].get("thinking_tokens") is not None]
print("thinking tokens: n",len(tt),"mean",sum(tt)/max(1,len(tt)),"max",max(tt) if tt else None)
PY
