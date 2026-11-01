#!/usr/bin/env bash
#SBATCH --job-name=glm53-serve
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
S=/gpfs/scratch/bsc21/bsc021885/cf; M=$S/models/GLM-5.3; E=$S/evalglm53; V=$S/envs/vllm32/bin
D=$S/data/contrast13m_200; P=$S/data/packs; CP=$P/probeforge_eval_candidate_2026-10-13; VX=$P/probeforge_coverage_extension_2026-10-13
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); HEAD=${NODES[0]}; HEAD_IP=$(getent hosts $HEAD | awk '{print $1}')
echo "nodes ${NODES[*]} head $HEAD $HEAD_IP $(date)"
COMMON="$M --tensor-parallel-size 4 --pipeline-parallel-size 4 --nnodes 4 --master-addr $HEAD_IP --master-port 29555 --max-model-len 16384 --max-num-seqs 128 --gpu-memory-utilization 0.90 --enforce-eager"
ENVS="unset PYTHONPATH PYTHONUSERBASE TRANSFORMERS_CACHE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMEXPR_MAX_THREADS=64 OMP_NUM_THREADS=8 FLASHINFER_CUBIN_DIR=$S/flashinfer_cubins VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0; export VLLM_HOST_IP=\$(getent hosts \$(hostname) | awk '{print \$1}' | head -1); echo host_ip \$VLLM_HOST_IP"
r=1; for n in "${NODES[@]:1}"; do
  srun --nodes=1 --ntasks=1 -w $n --exact --gres=gpu:4 --cpus-per-task=80 bash -c "$ENVS; $V/vllm serve $COMMON --node-rank $r --headless" > $E/worker_${r}_$SLURM_JOB_ID.log 2>&1 & r=$((r+1)); done
bash -c "$ENVS; $V/vllm serve $COMMON --node-rank 0 --served-model-name glm53 --reasoning-parser glm45 --port 8000" > $E/serve_$SLURM_JOB_ID.log 2>&1 &
SERVE=$!
for i in $(seq 1 170); do curl -s localhost:8000/v1/models 2>/dev/null | grep -q glm53 && break; kill -0 $SERVE 2>/dev/null || { echo "vllm serve exited"; break; }; sleep 30; done
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v1/models)" = 200 ] || { echo "server not up"; grep -vE "^\s*$" $E/serve_$SLURM_JOB_ID.log | tail -40; tail -15 $E/worker_1_$SLURM_JOB_ID.log; scancel $SLURM_JOB_ID; exit 1; }
echo "server up $(date)"
source ~/probeforge/slurm/environment.sh; cd ~/probeforge; export CF_API_TIMEOUT=1800
PIDS=()
wpids() { [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"; PIDS=(); }
go() { # name inputs extra budget nshards
  local name=$1 inp=$2 extra=$3 budget=$4 n=$5; local rem=$(python $E/remaining.py $name $inp $n)
  echo "set $name remaining $rem $(date)"; [ "$rem" = 0 ] && return
  for f in $E/${name}_r*.json; do k=${f##*_r}; k=${k%.json}
    contrastforge collect --backend api --base-url http://localhost:8000/v1 --model glm53 --inputs $inp --only-ids $f \
      --out $E/test_$name.r$SLURM_JOB_ID.$k.jsonl --budgets $budget --samples 1 --api-extra $E/$extra > $E/client_${name}_r$k.log 2>&1 &
    PIDS+=($!)
  done; }
# effort check: is reasoning_effort honoured? (one prompt, low vs max)
for eff in low max; do curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"glm53\",\"messages\":[{\"role\":\"user\",\"content\":\"A train covers 6051 km in 37 h. Exact speed in km/h as a fraction?\"}],\"max_tokens\":4096,\"temperature\":0,\"reasoning_effort\":\"$eff\",\"chat_template_kwargs\":{\"reasoning_effort\":\"$eff\"}}" | python3 -c "import sys,json;r=json.load(sys.stdin);m=r['choices'][0]['message'];print('EFFORT',sys.argv[1],'completion_tokens',r['usage']['completion_tokens'],'reasoning_chars',len(m.get('reasoning_content') or m.get('reasoning') or ''),'content',(m.get('content') or '')[:80].replace(chr(10),' '))" $eff; done
go main4k $D/test_eval12.jsonl extra_low.json 4096 64; wpids
go think26 $D/test_eval12.jsonl extra_max.json 4096 13
go think60 $CP/test_diagnostic60.jsonl extra_max.json 4096 26
go diag60 $CP/test_diagnostic60.jsonl extra_low.json 4096 100; wpids
go core60 $CP/test_core60.jsonl extra_low.json 4096 100; wpids
go dev112 $VX/development112_native.jsonl extra_low.json 4096 96; wpids
echo "all sets done $(date)"
