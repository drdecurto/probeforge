#!/usr/bin/env bash
#SBATCH --job-name=cf-think-eval
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=$S/evalthink; mkdir -p $E; cd ~/probeforge
run() { CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$2.out --error=$E/collect_$2.err \
  contrastforge collect --inputs $T --only-ids $D/think_ids.json --out $E/test_$2.jsonl --model $S/models/$3 $4 --budgets 4096 --samples 1 --thinking; }
run 0 base8b Qwen3-8B "" & run 1 mined8b Qwen3-8B "--adapter $S/runs/sftv2_mined_s1" & run 2 base14b Qwen3-14B "" & run 3 random14b Qwen3-14B "--adapter $S/runs/sft14b_random" &
wait
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalthink"
for n in ("base8b","mined8b","base14b","random14b"):
    recs=[json.loads(l) for l in open(f"{E}/test_{n}.jsonl")]
    ok=sum(r["verification"]["correct"] for r in recs); fin=sum(r["finish_reason"]=="stop" for r in recs)
    tt=[r["usage"].get("thinking_tokens") or 0 for r in recs]; sec=sum(r["generation_request_seconds"] for r in recs)/len(recs)
    print(f"THINK {n:10s} n={len(recs)} correct={ok} ({ok/len(recs):.2f}) finished={fin} mean_think_tokens={sum(tt)/len(tt):.0f} mean_s={sec:.0f}")
PY
echo "end $(date)"
