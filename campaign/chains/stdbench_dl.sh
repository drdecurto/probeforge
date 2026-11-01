S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; export HF_HOME=$S/hf_home; unset PYTHONPATH PYTHONUSERBASE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
for rid in utter-project/EuroLLM-9B-Instruct CohereLabs/aya-expanse-8b; do
  M=$S/models/$(basename $rid); rm -f "$M/.cache/huggingface/download/*.pth.lock"
  for i in $(seq 1 80); do HF_HUB_ENABLE_HF_TRANSFER=1 timeout 280 $S/envs/vllm32/bin/hf download $rid --local-dir $M --exclude "original/*" --max-workers 8 >/dev/null 2>$L/stdbench_dl.err && { echo "$rid complete $(du -sh $M|cut -f1) $(date +%T)"; break; }; grep -qiE "gated|403|401|restricted|not found|denied|approval" $L/stdbench_dl.err && { echo "$rid FAILED: $(grep -iE "error" $L/stdbench_dl.err|tail -1|cut -c1-200)"; break; }; sleep 3; done
done
echo "DONE $(date)"
