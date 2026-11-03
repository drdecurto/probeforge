import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter, out = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload(); m.save_pretrained(out, max_shard_size="5GB"); AutoTokenizer.from_pretrained(base).save_pretrained(out); print("merged", out)
