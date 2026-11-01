S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; export HF_HOME=$S/hf_home; unset PYTHONPATH PYTHONUSERBASE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
for R in zai-org/GLM-4.7-Flash zai-org/GLM-5.3; do M=$S/models/$(basename $R); mkdir -p $M
for i in $(seq 1 400); do HF_HUB_ENABLE_HF_TRANSFER=1 timeout 280 $S/envs/vllm32/bin/hf download $R --local-dir $M --max-workers 8 >/dev/null 2>$L/dlglm.err && { echo "$R complete $(du -sh $M | cut -f1) $(date +%T)"; break; }; sleep 3; [ $((i % 10)) = 0 ] && echo "$(date +%T) $R loop $i: $(du -sh $M | cut -f1)"; done; done
echo "DONE $(date)"
