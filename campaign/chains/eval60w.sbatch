#!/usr/bin/env bash
#SBATCH --job-name=cf-eval60w
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
S=/gpfs/scratch/bsc21/bsc021885/cf; T=$S/data/contrast60/test.jsonl; E=$S/eval60w; mkdir -p $E
cd ~/probeforge
echo "node $(hostname) start $(date)"
run() { # $1 gpu, $2 name, $3 adapter-arg
  CUDA_VISIBLE_DEVICES=$1 srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none \
    --output=$E/collect_$2.out --error=$E/collect_$2.err \
    contrastforge collect --inputs $T --out $E/test_$2.jsonl --model $S/models/Qwen3-8B $3 --budgets 512 --samples 1
}
run 0 base "" & run 1 translated "--adapter $S/runs/sft60w_translated" & run 2 random "--adapter $S/runs/sft60w_random" & run 3 mined "--adapter $S/runs/sft60w_mined" &
wait
run 0 no_counterfactual "--adapter $S/runs/sft60w_no_counterfactual"
for n in base translated random mined no_counterfactual; do
  contrastforge evaluate --inputs $T --rollouts $E/test_$n.jsonl --out $E/report_$n.json > /dev/null && echo "report_$n ok"
done
contrastforge compare --reports $E/report_base.json $E/report_translated.json $E/report_random.json $E/report_mined.json $E/report_no_counterfactual.json --out $E/comparison.json | head -60
echo "end $(date)"
