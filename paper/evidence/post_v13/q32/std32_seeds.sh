#!/usr/bin/env bash
S=/gpfs/scratch/bsc21/bsc021885/cf; Q=$S/q32; SD=$S/stdbench; LE=$S/envs/lmeval/bin/lm_eval
for n in random_s2 random_s3 v4c_s2 v4c_s3; do M=$S/models/merged/q32-$n; [ -f $M/config.json ] || ( source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1; python $Q/merge32.py $S/models/Qwen3-32B $S/runs/sft32b_$n $M < /dev/null > $Q/logs/merge_$n.log 2>&1 ); echo "merge $n $([ -f $M/config.json ] && echo ok || echo FAIL)"; done
unset PYTHONPATH PYTHONUSERBASE; export NUMEXPR_MAX_THREADS=64 HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
MC=$(grep '^MC=' $SD/node_std.sh | cut -d'"' -f2); MG=$(grep '^MG=' $SD/node_std.sh | cut -d'"' -f2)
for spec in "qwen3-32b-random_s2|merged/q32-random_s2" "qwen3-32b-random_s3|merged/q32-random_s3" "qwen3-32b-v4c_s2|merged/q32-v4c_s2" "qwen3-32b-v4c_s3|merged/q32-v4c_s3"; do IFS='|' read name path <<< "$spec"
  for part in mc mgsm; do O=$SD/out/$name/$part; [ -f $O/DONE ] && continue; mkdir -p $O
    if [ $part = mc ]; then T=$MC; X=""; else T=$MG; X="--num_fewshot 8 --log_samples"; fi
    $LE --model vllm --model_args "pretrained=$S/models/$path,dtype=bfloat16,tensor_parallel_size=4,gpu_memory_utilization=0.85,max_model_len=4096,enable_thinking=False,seed=1234" --tasks $T $X --batch_size auto --seed 1234 --output_path $O > $SD/logs/${name}_$part.log 2>&1 && touch $O/DONE
    echo "$name $part $([ -f $O/DONE ] && echo ok || echo FAIL) $(date +%T)"; done; done
