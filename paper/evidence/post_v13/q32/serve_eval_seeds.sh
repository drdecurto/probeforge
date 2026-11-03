#!/usr/bin/env bash
S=/gpfs/scratch/bsc21/bsc021885/cf; Q=$S/q32; B=$S/energybench; D=$S/data/contrast13m_200; V=$S/envs/vllm32/bin
unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMEXPR_MAX_THREADS=64
$V/vllm serve $S/models/Qwen3-32B --served-model-name base32b --tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.88 \
  --enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules random32b_s2=$S/runs/sft32b_random_s2 random32b_s3=$S/runs/sft32b_random_s3 v4c32b_s2=$S/runs/sft32b_v4c_s2 v4c32b_s3=$S/runs/sft32b_v4c_s3 --port 8200 > $Q/logs/serve_$SLURM_JOB_ID.log 2>&1 &
SP=$!; for i in $(seq 1 90); do curl -s localhost:8200/v1/models | grep -q v4c32b_s3 && break; kill -0 $SP || break; sleep 10; done
curl -s localhost:8200/v1/models | grep -q v4c32b_s3 || { echo "server failed"; tail -20 $Q/logs/serve_$SLURM_JOB_ID.log; exit 1; }
echo "server up $(date)"
# main benchmark (API backend, resumable shard files)
source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; cd ~/probeforge; export CF_API_TIMEOUT=600
for m in random32b_s2 random32b_s3 v4c32b_s2 v4c32b_s3; do rem=$(python $Q/remaining.py $m $D/test_eval12.jsonl 16); echo "main $m remaining $rem"
  for f in $Q/eval/${m}_r*.json; do [ -f "$f" ] || continue; k=${f##*_r}; k=${k%.json}
    contrastforge collect --backend api --base-url http://localhost:8200/v1 --model $m --inputs $D/test_eval12.jsonl --only-ids $f --api-extra $Q/api_extra.json \
      --out $Q/eval/test_$m.r$SLURM_JOB_ID.$k.jsonl --budgets 512 --samples 1 < /dev/null > $Q/logs/collect_${m}_$k.log 2>&1 & done; done
# energy benchmark: every (model, protocol, language) as its own client, CPU-only MiniLM
for m in random32b_s2 random32b_s3 v4c32b_s2 v4c32b_s3; do for proto in out out_chat; do PY=$B/eval_patched.py; [ $proto = out_chat ] && PY=$B/eval_chat.py; mkdir -p $Q/energy/$proto/$m
  for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $Q/energy/$proto/$m/$l.json ] && continue; W=$Q/energy/run/$proto/$m/$l; rm -rf $W; mkdir -p $W; cp $B/langs/$l/problems.json $W/
    ( cd $W && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" CHAT_TOKENIZER=$S/models/Qwen3-32B API_URL=http://localhost:8200/v1/completions OUT_JSON=$Q/energy/$proto/$m/$l.json.tmp $V/python $PY $m 0 > $Q/logs/en_${proto}_${m}_$l.log 2>&1 && mv $Q/energy/$proto/$m/$l.json.tmp $Q/energy/$proto/$m/$l.json ) &
  done; done; done
wait; kill $SP; echo "end $(date)"
