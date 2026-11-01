#!/usr/bin/env bash
#SBATCH --job-name=cf-evalv2
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test.jsonl; E=$S/evalv2; mkdir -p $E; cd ~/probeforge
echo "node $(hostname) start $(date)"
run() { CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none \
    --output=$E/collect_$2.out --error=$E/collect_$2.err \
    contrastforge collect --inputs $T --only-ids $D/eval_ids.json --out $E/test_$2.jsonl --model $S/models/Qwen3-8B $3 --budgets 512 --samples 1; }
run 0 base "" & run 1 translated "--adapter $S/runs/sftv2_translated" & run 2 random "--adapter $S/runs/sftv2_random" & run 3 mined "--adapter $S/runs/sftv2_mined" &
wait
echo "no_counterfactual deferred"
for n in base translated random mined; do contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report_$n.json >/dev/null && echo "report_$n ok"; done
contrastforge compare --reports $E/report_base.json $E/report_translated.json $E/report_random.json $E/report_mined.json --out $E/comparison.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/evalv2"
for n in ("base","translated","random","mined"):
    b=json.load(open(f"{E}/report_{n}.json"))["budgets"]["512"]
    print(f"{n:18s} pass@1={b['pass_at_1']:.3f} all_views={b['all_views_correct']:.3f} tok={b['mean_completion_tokens']:.0f}")
    print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())}); print("   view",{k:round(v,2) for k,v in b["per_view"].items()})
for x in json.load(open(f"{E}/comparison.json"))["comparisons"]: print(x["treatment"].split("/")[-1],round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]],"groups",x["groups"])
PY
echo "end $(date)"
