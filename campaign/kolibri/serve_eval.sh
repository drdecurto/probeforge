#!/usr/bin/env bash
export VLLM_CACHE_ROOT=/gpfs/scratch/bsc21/bsc021885/cf/vllm_cache
S=/gpfs/scratch/bsc21/bsc021885/cf; G=$S/kolibri; B=$S/energybench; D=$S/data/contrast13m_200; V=$S/envs/kolvllm/bin
MC="belebele_eng_Latn,belebele_spa_Latn,belebele_cat_Latn,belebele_arb_Arab,belebele_eus_Latn,belebele_fra_Latn,belebele_por_Latn,belebele_ita_Latn,belebele_deu_Latn,belebele_rus_Cyrl,belebele_zho_Hans,belebele_zho_Hant,global_mmlu_ar,global_mmlu_de,global_mmlu_en,global_mmlu_es,global_mmlu_fr,global_mmlu_it,global_mmlu_pt,global_mmlu_zh"
MG="mgsm_native_cot_de,mgsm_native_cot_en,mgsm_native_cot_es,mgsm_native_cot_fr,mgsm_native_cot_ru,mgsm_native_cot_zh"
for spec in $KSPECS; do IFS='|' read m path <<< "$spec"
  emiss=0; for p in out out_chat; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $G/energy/$p/$m/$l.json ] || emiss=$((emiss+1)); done; done
  rem=$(bash -c "source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; python $G/remaining.py $m $D/test_eval12.jsonl 16"); echo "$m main remaining $rem energy missing $emiss $(date +%T)"
  smiss=0; for part in mc mgsm; do [ -f $S/stdbench/out/kol-$m/$part/DONE ] || smiss=$((smiss+1)); done; echo "$m std missing $smiss"
  [ "$rem" = 0 ] && [ $emiss = 0 ] && [ $smiss = 0 ] && continue
  ( unset PYTHONPATH PYTHONUSERBASE TRANSFORMERS_CACHE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMEXPR_MAX_THREADS=64
    exec $V/vllm serve $path --served-model-name $m --tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.88 --port 8400 ) > $G/logs/serve_${m}_$SLURM_JOB_ID.log 2>&1 &
  SP=$!; for i in $(seq 1 270); do curl -s localhost:8400/v1/models | grep -q "\"$m\"" && break; kill -0 $SP || break; sleep 10; done
  curl -s localhost:8400/v1/models | grep -q "\"$m\"" || { echo "server $m failed"; tail -20 $G/logs/serve_${m}_$SLURM_JOB_ID.log; kill $SP; continue; }
  ( source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; cd ~/probeforge; export CF_API_TIMEOUT=600
    for f in $G/eval/${m}_r*.json; do [ -f "$f" ] || continue; k=${f##*_r}; k=${k%.json}
      contrastforge collect --backend api --base-url http://localhost:8400/v1 --model $m --inputs $D/test_eval12.jsonl --only-ids $f --api-extra $G/api_extra.json \
        --out $G/eval/test_$m.r$SLURM_JOB_ID.$k.jsonl --budgets 512 --samples 1 < /dev/null > $G/logs/collect_${m}_$k.log 2>&1 & done; wait ) &
  for proto in out out_chat; do PY=$B/eval_patched.py; [ $proto = out_chat ] && PY=$B/eval_chat.py; mkdir -p $G/energy/$proto/$m
    for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $G/energy/$proto/$m/$l.json ] && continue; W=$G/energy/run/$proto/$m/$l; rm -rf $W; mkdir -p $W; cp $B/langs/$l/problems.json $W/
      ( cd $W && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" CHAT_TOKENIZER=$S/models/Kolibri-1-BF16 API_URL=http://localhost:8400/v1/completions OUT_JSON=$G/energy/$proto/$m/$l.json.tmp $S/envs/vllm32/bin/python $PY $m 0 > $G/logs/en_${proto}_${m}_$l.log 2>&1 && mv $G/energy/$proto/$m/$l.json.tmp $G/energy/$proto/$m/$l.json ) &
    done; done
  for part in mc mgsm; do O=$S/stdbench/out/kol-$m/$part; [ -f $O/DONE ] && continue; mkdir -p $O
    if [ $part = mc ]; then T=$MC; X=""; else T=$MG; X="--num_fewshot 8 --log_samples"; fi
    ( unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
      timeout 6000 $S/envs/lmeval/bin/lm_eval --model local-completions --model_args "model=$m,base_url=http://localhost:8400/v1/completions,tokenizer=$S/models/Kolibri-1-BF16,num_concurrent=16,max_retries=3,max_length=4096,seed=1234" \
        --tasks $T $X --batch_size 1 --seed 1234 --output_path $O > $G/logs/std_${m}_$part.log 2>&1 && touch $O/DONE; echo "$m std $part $([ -f $O/DONE ] && echo ok || echo FAIL) $(date +%T)" ) &
  done
  wait $(jobs -p | grep -v "^$SP$") 2>/dev/null; sleep 5; kill $SP; wait $SP 2>/dev/null; echo "$m done $(date +%T)"
done
