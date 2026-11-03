#!/usr/bin/env bash
S=/gpfs/scratch/bsc21/bsc021885/cf; G=$S/glm; B=$S/energybench; D=$S/data/contrast13m_200; V=$S/envs/vllm32/bin
for n in random_s2 random_s3 v4c_s2 v4c_s3; do M=$G/merged/glm-$n; [ -f $M/config.json ] || ( source $G/env.sh; python $G/merge.py $S/models/GLM-4.7-Flash $S/runs/sftglm_$n $M < /dev/null > $G/logs/merge_$n.log 2>&1 ); echo "merge $n $([ -f $M/config.json ] && echo ok || echo FAIL)"; done
for spec in "glmrandom_s2|$G/merged/glm-random_s2" "glmrandom_s3|$G/merged/glm-random_s3" "glmv4c_s2|$G/merged/glm-v4c_s2" "glmv4c_s3|$G/merged/glm-v4c_s3"; do IFS='|' read m path <<< "$spec"
  emiss=0; for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $G/energy/$p/$m/$l.json ] || emiss=$((emiss+1)); done; done
  rem=$(bash -c "source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; python $G/remaining.py $m $D/test_eval12.jsonl 16"); echo "$m main remaining $rem energy missing $emiss $(date +%T)"
  [ "$rem" = 0 ] && [ $emiss = 0 ] && continue
  ( unset PYTHONPATH PYTHONUSERBASE TRANSFORMERS_CACHE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMEXPR_MAX_THREADS=64
    exec $V/vllm serve $path --served-model-name $m --tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.88 --port 8300 ) > $G/logs/serve_${m}_$SLURM_JOB_ID.log 2>&1 &
  SP=$!; for i in $(seq 1 90); do curl -s localhost:8300/v1/models | grep -q "\"$m\"" && break; kill -0 $SP || break; sleep 10; done
  curl -s localhost:8300/v1/models | grep -q "\"$m\"" || { echo "server $m failed"; tail -20 $G/logs/serve_${m}_$SLURM_JOB_ID.log; kill $SP; continue; }
  ( source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; cd ~/probeforge; export CF_API_TIMEOUT=600
    for f in $G/eval/${m}_r*.json; do [ -f "$f" ] || continue; k=${f##*_r}; k=${k%.json}
      contrastforge collect --backend api --base-url http://localhost:8300/v1 --model $m --inputs $D/test_eval12.jsonl --only-ids $f --api-extra $G/api_extra.json \
        --out $G/eval/test_$m.r$SLURM_JOB_ID.$k.jsonl --budgets 512 --samples 1 < /dev/null > $G/logs/collect_${m}_$k.log 2>&1 & done; wait ) &
  for proto in out out_chat; do PY=$B/eval_patched.py; [ $proto = out_chat ] && PY=$B/eval_chat.py; mkdir -p $G/energy/$proto/$m
    for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $G/energy/$proto/$m/$l.json ] && continue; W=$G/energy/run/$proto/$m/$l; rm -rf $W; mkdir -p $W; cp $B/langs/$l/problems.json $W/
      ( cd $W && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" CHAT_TOKENIZER=$S/models/GLM-4.7-Flash API_URL=http://localhost:8300/v1/completions OUT_JSON=$G/energy/$proto/$m/$l.json.tmp $V/python $PY $m 0 > $G/logs/en_${proto}_${m}_$l.log 2>&1 && mv $G/energy/$proto/$m/$l.json.tmp $G/energy/$proto/$m/$l.json ) &
    done; done
  wait $(jobs -p | grep -v "^$SP$") 2>/dev/null; sleep 5; kill $SP; wait $SP 2>/dev/null; echo "$m done $(date +%T)"
done
