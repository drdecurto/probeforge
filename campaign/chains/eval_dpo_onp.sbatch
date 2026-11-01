#!/usr/bin/env bash
#SBATCH --job-name=cf-eval-dpo-onp
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=$S/evalv2; cd ~/probeforge
for K in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_dpoonp_s$K.out --error=$E/collect_dpoonp_s$K.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$K.json --out $E/test_dpo_onp.shard$K.jsonl --model $S/models/Qwen3-8B --adapter $S/runs/dpo_onp --budgets 512 --samples 1 & done; wait
cat $E/test_dpo_onp.shard*.jsonl > $E/test_dpo_onp.jsonl; cp $E/test_dpo_onp.shard0.jsonl.manifest.json $E/test_dpo_onp.jsonl.manifest.json
contrastforge evaluate --inputs $T --rollouts $E/test_dpo_onp.jsonl --out $E/report12_dpo_onp.json >/dev/null
contrastforge compare --reports $E/report12_mined_s1.json $E/report12_dpo_onp.json --out $E/comparison12_dpo_onp.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalv2"
for n in ("mined_s1","dpo_onp"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]; print(f"DPO {n:9s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} base_view={b['per_view']['base']:.2f} tok={b['mean_completion_tokens']:.0f}")
for x in json.load(open(f"{E}/comparison12_dpo_onp.json"))["comparisons"]: print("dpo vs sft:",round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
echo "end $(date)"
