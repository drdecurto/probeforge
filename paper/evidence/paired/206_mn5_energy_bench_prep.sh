S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; X=$S/ext/energy_multilingual_llm_evaluation; B=$S/energybench; mkdir -p $B/langs $B/out
unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home; unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
V=$S/envs/vllm32/bin
# 1) per-language problems.json (EN from problems_actual.json, others from the v1.1 dataset modules)
$V/python - <<'PY'
import json, runpy, os
X="/gpfs/scratch/bsc21/bsc021885/cf/ext/energy_multilingual_llm_evaluation"; B="/gpfs/scratch/bsc21/bsc021885/cf/energybench"
langs={"en":None}; langs.update({l:f"{X}/datasets/all_problems_dataset_{l}_v1.1_decurto_and_dezarza.py" for l in ["ar","ca","de","es","eu","fr","it","lb","pt","ru","zh-cn","zh-tw"]})
for l,p in langs.items():
    probs=json.load(open(f"{X}/src_gpu/problems_actual.json")) if p is None else runpy.run_path(p)["all_problems"]
    os.makedirs(f"{B}/langs/{l}",exist_ok=True); json.dump(probs,open(f"{B}/langs/{l}/problems.json","w"),ensure_ascii=False,indent=1)
    print(l,len(probs),end="; ")
print()
PY
# 2) evaluator copy: identical prompt/decoding/scoring; API URL from env; extra dump of solutions for secondary scoring
python3 - <<'PY'
X="/gpfs/scratch/bsc21/bsc021885/cf/ext/energy_multilingual_llm_evaluation"; B="/gpfs/scratch/bsc21/bsc021885/cf/energybench"
s=open(f"{X}/src_gpu/llm_custom_vllm_already_loaded_energy.py").read()
a='ImprovedReasoningEvaluator(MODEL, "http://localhost:8000/v1/completions")'; assert a in s
s=s.replace(a,'ImprovedReasoningEvaluator(MODEL, os.environ.get("API_URL","http://localhost:8000/v1/completions"))')
b='    print(json.dumps(analysis, indent=2))'; assert b in s
s=s.replace(b,b+'''
    if os.environ.get("OUT_JSON"):
        with open(os.environ["OUT_JSON"], "w") as _f:
            json.dump({"analysis": analysis, "problems": problems, "solutions": [e["solutions"] for e in evaluations],
                       "average_scores": [float(e["average_score"]) for e in evaluations]}, _f, ensure_ascii=False, default=float)''')
open(f"{B}/eval_patched.py","w").write(s); print("eval_patched.py written (2 additive edits)")
PY
diff <(sed 's/os.environ.get("API_URL","http:\/\/localhost:8000\/v1\/completions")/"http:\/\/localhost:8000\/v1\/completions"/' $B/eval_patched.py) $X/src_gpu/llm_custom_vllm_already_loaded_energy.py | head -12
# 3) deps + embedding model + anchor model
for i in 1 2 3 4 5; do timeout 280 $V/pip install -q "sentence-transformers>=2.6.1" seaborn scikit-learn matplotlib pynvml psutil 2>&1 | tail -1 && break; done
$V/python -c "import sentence_transformers, seaborn, sklearn, pynvml; print('deps ok', sentence_transformers.__version__)"
$V/python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); print('MiniLM ok', m.encode(['x']).shape)"
for i in $(seq 1 20); do timeout 280 $V/hf download microsoft/Phi-4-mini-instruct --local-dir $S/models/Phi-4-mini-instruct >/dev/null 2>&1 && break; done
du -sh $S/models/Phi-4-mini-instruct; ls $S/models/Phi-4-mini-instruct | head -4
$V/python -c "
from vllm.model_executor.models.registry import ModelRegistry as R; a=R.get_supported_archs(); print('Qwen3', 'Qwen3ForCausalLM' in a, 'Phi3', 'Phi3ForCausalLM' in a)"
ls $S/runs | grep -E "^sftv2_mined_s1$|^sft14b_random$|^sftv3_14b_mined$|^sft14b_distill$|^sftphi4_mined$"
