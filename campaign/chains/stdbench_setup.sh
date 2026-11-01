S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; export HF_HOME=$S/hf_home; unset PYTHONPATH PYTHONUSERBASE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
echo "START $(date)"
# 1) lm-eval venv layered on vllm32 (vllm32 untouched)
if [ ! -x $S/envs/lmeval/bin/lm_eval ]; then
  $S/envs/vllm32/bin/python -m venv $S/envs/lmeval
  SP=$($S/envs/vllm32/bin/python -c "import site;print(site.getsitepackages()[0])"); MY=$($S/envs/lmeval/bin/python -c "import site;print(site.getsitepackages()[0])")
  echo "$SP" > $MY/vllm32.pth
  timeout 280 $S/envs/lmeval/bin/pip install -q "lm-eval[math]==0.4.9.1" langdetect immutabledict 2>&1 | tail -3
fi
$S/envs/lmeval/bin/python -c "import lm_eval,vllm,transformers,datasets;print('lm_eval',lm_eval.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__,'datasets',datasets.__version__)"
# 2) list candidate tasks
timeout 280 $S/envs/lmeval/bin/python - <<'PY' > $S/stdbench/tasklist.txt
from lm_eval.tasks import TaskManager
tm=TaskManager(); names=tm.all_tasks
for n in names:
    if n.startswith(("mgsm_native_cot_","mgsm_direct_","belebele_","global_mmlu_")) : print(n)
PY
wc -l $S/stdbench/tasklist.txt; grep -E "^global_mmlu" $S/stdbench/tasklist.txt | head -30 | tr '\n' ' '; echo
# 3) comparator models
for rid in CohereLabs/aya-expanse-8b meta-llama/Llama-3.1-8B-Instruct google/gemma-3-12b-it BSC-LT/salamandra-7b-instruct Qwen/Qwen2.5-7B-Instruct; do
  M=$S/models/$(basename $rid)
  for i in $(seq 1 60); do HF_HUB_ENABLE_HF_TRANSFER=1 timeout 280 $S/envs/vllm32/bin/hf download $rid --local-dir $M --exclude "original/*" "*.pth" --max-workers 8 >/dev/null 2>$L/stdbench_dl.err && { echo "$rid complete $(du -sh $M|cut -f1) $(date +%T)"; break; }; grep -qiE "gated|403|401|restricted" $L/stdbench_dl.err && { echo "$rid ACCESS DENIED: $(tail -1 $L/stdbench_dl.err|cut -c1-160)"; break; }; sleep 3; done
done
echo "DONE $(date)"
