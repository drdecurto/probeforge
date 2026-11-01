#!/usr/bin/env bash
#SBATCH --job-name=cf-eval14b-ad
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast13m_200; T=$D/test_eval12.jsonl; E=$S/eval14b; cd ~/probeforge
gpu() { for n in $2; do CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_${n}_s$3.out --error=$E/collect_${n}_s$3.err \
  contrastforge collect --inputs $T --only-ids $D/eshard$3.json --out $E/test_$n.shard$3.jsonl --model $S/models/Qwen3-14B --adapter $S/runs/sft14b_$n --budgets 512 --samples 1; done; }
gpu 0 "random translated" 0 & gpu 1 "random translated" 1 & gpu 2 "random mined" 2 & gpu 3 "random mined" 3 &
wait
gpu 0 "mined" 0 & gpu 1 "mined" 1 & gpu 2 "translated" 2 & gpu 3 "translated" 3 &
wait
for n in random mined translated; do cat $E/test_$n.shard*.jsonl > $E/test_$n.jsonl; cp $E/test_$n.shard0.jsonl.manifest.json $E/test_$n.jsonl.manifest.json; contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report12_$n.json >/dev/null && echo "report12_$n ok"; done
contrastforge compare --reports $E/report12_base14b.json $E/report12_translated.json $E/report12_random.json $E/report12_mined.json --out $E/comparison12.json >/dev/null
python - <<'PY'
import json
E="/gpfs/scratch/bsc21/bsc021885/cf/eval14b"
for n in ("base14b","translated","random","mined"):
    b=json.load(open(f"{E}/report12_{n}.json"))["budgets"]["512"]
    print(f"14B {n:11s} pass@1={b['pass_at_1']:.3f} worst={b['worst_language']:.3f} boundary={b['per_view']['boundary']:.2f} base_view={b['per_view']['base']:.2f} cf={b['per_view']['counterfactual']:.2f}"); print("   lang",{k:round(v,2) for k,v in sorted(b["per_language"].items())})
for x in json.load(open(f"{E}/comparison12.json"))["comparisons"]: print(x["treatment"].split("/")[-1],round(x["mean_paired_group_difference"],3),[round(v,3) for v in x["paired_ci95"]])
PY
echo "end $(date)"
