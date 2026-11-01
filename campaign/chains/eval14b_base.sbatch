#!/usr/bin/env bash
#SBATCH --job-name=cf-eval14b-base
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
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=$S/eval14b; mkdir -p $E; cd ~/probeforge
echo "node $(hostname) start $(date)"
for K in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_base$K.out --error=$E/collect_base$K.err \
    contrastforge collect --inputs $T --only-ids $D/eshard$K.json --out $E/test_base.shard$K.jsonl --model $S/models/Qwen3-14B --budgets 512 --samples 1 &
done
wait
cat $E/test_base.shard*.jsonl > $E/test_base.jsonl; cp $E/test_base.shard0.jsonl.manifest.json $E/test_base.jsonl.manifest.json
contrastforge evaluate --inputs $T --rollouts $E/test_base.jsonl --out $E/report12_base14b.json >/dev/null && echo "report ok"
python - <<'PY'
import json
b=json.load(open("/gpfs/scratch/bsc21/bsc021885/cf/eval14b/report12_base14b.json"))["budgets"]["512"]
print("Qwen3-14B base pass@1",round(b["pass_at_1"],3),"worst",round(b["worst_language"],3)); print(" lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())}); print(" view",{k:round(v,2) for k,v in b["per_view"].items()})
PY
echo "end $(date)"
