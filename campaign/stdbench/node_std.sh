#!/usr/bin/env bash
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/stdbench; LE=$S/envs/lmeval/bin/lm_eval
unset PYTHONPATH PYTHONUSERBASE; export NUMEXPR_MAX_THREADS=64 HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
MC="belebele_eng_Latn,belebele_spa_Latn,belebele_cat_Latn,belebele_arb_Arab,belebele_eus_Latn,belebele_fra_Latn,belebele_por_Latn,belebele_ita_Latn,belebele_deu_Latn,belebele_rus_Cyrl,belebele_zho_Hans,belebele_zho_Hant,global_mmlu_ar,global_mmlu_de,global_mmlu_en,global_mmlu_es,global_mmlu_fr,global_mmlu_it,global_mmlu_pt,global_mmlu_zh"
MG="mgsm_native_cot_de,mgsm_native_cot_en,mgsm_native_cot_es,mgsm_native_cot_fr,mgsm_native_cot_ru,mgsm_native_cot_zh"
C=$D/claims/$SLURM_JOB_ID; mkdir -p $C; H=$(hostname -s)
run() { # name base adapter part
  local name=$1 base=$2 adapter=$3 part=$4 O=$D/out/$1/$4
  [ -f $O/DONE ] && return 0; mkdir -p $O
  A="pretrained=$S/models/$base,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,enable_thinking=False,seed=1234"
  [ -n "$adapter" ] && A="$A,lora_local_path=$S/runs/$adapter,max_lora_rank=16"
  if [ $part = mc ]; then T=$MC; X=""; else T=$MG; X="--num_fewshot 8 --log_samples"; fi
  CUDA_VISIBLE_DEVICES=$k timeout 6600 $LE --model vllm --model_args "$A" --tasks $T $X --batch_size auto --seed 1234 --output_path $O > $D/logs/${name}_$part.log 2>&1 && touch $O/DONE
  echo "$name $part $([ -f $O/DONE ] && echo ok || echo FAIL) $H gpu$k $(date +%T)"; [ -f $O/DONE ] || tail -3 $D/logs/${name}_$part.log
}
gpu_loop() { k=$1
  while true; do task=""
    while IFS='|' read name base adapter; do for part in mc mgsm; do [ -f $D/out/$name/$part/DONE ] && continue; mkdir $C/${name}__$part 2>/dev/null && { task="$name|$base|$adapter|$part"; break 2; }; done; done < $D/models.txt
    [ -z "$task" ] && break; IFS='|' read name base adapter part <<< "$task"; run $name $base "$adapter" $part
  done; echo "gpu $H/$k idle $(date +%T)"; }
for k in 0 1 2 3; do gpu_loop $k & sleep 3; done; wait
