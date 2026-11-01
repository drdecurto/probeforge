#!/usr/bin/env bash
#SBATCH --job-name=muse-serve
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
set -u
S=/gpfs/scratch/bsc21/bsc021885/cf; C=$S/containers; M=$S/models/muse-spark-nvfp4; E=$S/evalmuse; SIF=$C/vllm-muse.sif
D=$S/data/contrast13m_200; P=$S/data/packs; CP=$P/probeforge_eval_candidate_2026-10-13; V=$P/probeforge_coverage_extension_2026-10-13
module load singularity 2>/dev/null
R=$S/envs/ray_overlay; export SINGULARITYENV_PYTHONPATH=$R SINGULARITYENV_PREPEND_PATH=$R/bin
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_RUST_FRONTEND=0 SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_VLLM_USE_RUST_FRONTEND=0
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); HEAD=${NODES[0]}; HEAD_IP=$(getent hosts $HEAD | awk '{print $1}')
echo "nodes ${NODES[*]} head $HEAD $HEAD_IP $(date)"
RUN="singularity exec --nv -B /gpfs/scratch/bsc21/bsc021885 $SIF"
$RUN ray start --head --node-ip-address=$HEAD_IP --port=6379 --num-gpus=4 --block > $E/ray_head.log 2>&1 &
sleep 25
for n in "${NODES[@]:1}"; do IP=$(getent hosts $n | awk '{print $1}')
  srun --nodes=1 --ntasks=1 -w $n --exact --gres=gpu:4 --cpus-per-task=80 $RUN ray start --address=$HEAD_IP:6379 --node-ip-address=$IP --num-gpus=4 --block > $E/ray_$n.log 2>&1 & done
sleep 40; $RUN ray status 2>&1 | grep -E "GPU" | head -2
$RUN vllm serve $M --served-model-name muse --tensor-parallel-size 4 --pipeline-parallel-size 4 --distributed-executor-backend ray \
  --max-model-len 16384 --max-num-seqs 128 --gpu-memory-utilization 0.90 --reasoning-parser muse --language-model-only --trust-remote-code --moe-backend marlin \
  --port 8000 > $E/serve_$SLURM_JOB_ID.log 2>&1 &
SERVE=$!
for i in $(seq 1 120); do curl -s localhost:8000/v1/models 2>/dev/null | grep -q muse && break; kill -0 $SERVE 2>/dev/null || { echo "vllm serve exited"; break; }; sleep 30; done
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v1/models)" = 200 ] || { echo "server not up"; grep -vE "^\s*$" $E/serve_$SLURM_JOB_ID.log | tail -40; scancel $SLURM_JOB_ID; exit 1; }
echo "server up $(date)"
source ~/probeforge/slurm/environment.sh; cd ~/probeforge; export CF_API_TIMEOUT=1800
PIDS=()
wpids() { [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"; PIDS=(); }
go() { # name inputs extra budget nshards
  local name=$1 inp=$2 extra=$3 budget=$4 n=$5; local rem=$(python $E/remaining.py $name $inp $n)
  echo "set $name remaining $rem $(date)"; [ "$rem" = 0 ] && return
  for f in $E/${name}_r*.json; do k=${f##*_r}; k=${k%.json}
    contrastforge collect --backend api --base-url http://localhost:8000/v1 --model muse --inputs $inp --only-ids $f \
      --out $E/test_$name.r$SLURM_JOB_ID.$k.jsonl --budgets $budget --samples 1 --api-extra $E/$extra > $E/client_${name}_r$k.log 2>&1 &
    PIDS+=($!)
  done; }
# effort check: is reasoning_effort honoured? (one prompt, low vs xhigh)
for eff in low xhigh; do curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"muse\",\"messages\":[{\"role\":\"user\",\"content\":\"A train covers 6051 km in 37 h. Exact speed in km/h as a fraction?\"}],\"max_tokens\":4096,\"temperature\":0,\"reasoning_effort\":\"$eff\",\"chat_template_kwargs\":{\"reasoning_effort\":\"$eff\"}}" | python3 -c "import sys,json;r=json.load(sys.stdin);m=r['choices'][0]['message'];print('EFFORT',sys.argv[1],'completion_tokens',r['usage']['completion_tokens'],'reasoning_chars',len(m.get('reasoning_content') or m.get('reasoning') or ''),'content',(m.get('content') or '')[:80].replace(chr(10),' '))" $eff; done
go think26 $D/test_eval12.jsonl extra_xhigh.json 4096 13
go think60 $CP/test_diagnostic60.jsonl extra_xhigh.json 4096 26
go main4k $D/test_eval12.jsonl extra_low.json 4096 64; wpids
go dev112 $V/development112_native.jsonl extra_low.json 4096 96; wpids
go diag60 $CP/test_diagnostic60.jsonl extra_low.json 4096 100; wpids
go core60 $CP/test_core60.jsonl extra_low.json 4096 100; wpids
echo "all sets done $(date)"
