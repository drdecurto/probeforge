S=/gpfs/scratch/bsc21/bsc021885/cf
source ~/probeforge/slurm/environment.sh; unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
echo "START $(date)"
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("microsoft/phi-4", local_dir="/gpfs/scratch/bsc21/bsc021885/cf/models/phi-4",
                      allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.py"], max_workers=4)
print("downloaded to", p)
PY
python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("/gpfs/scratch/bsc21/bsc021885/cf/models/phi-4")
m = [{"role":"system","content":"SYS"},{"role":"user","content":"USER"}]
print("template OK:", repr(t.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)))
print("eos:", t.eos_token, "pad:", t.pad_token)
PY
du -sh /gpfs/scratch/bsc21/bsc021885/cf/models/phi-4; echo "DONE $(date)"
