import sys, json, random
sys.path.insert(0, "/gpfs/scratch/bsc21/bsc021885/cf/v4")
import gsm_ml as g
from datasets import load_dataset
from vllm import LLM, SamplingParams
ds = load_dataset("openai/gsm8k", "main")["train"]
idx = list(range(len(ds))); random.Random(20261022).shuffle(idx); idx = idx[:30]
llm = LLM(model=sys.argv[1], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, seed=1234); tok = llm.get_tokenizer()
chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
for lang in ("it", "pt"):
    tr = llm.generate([chat([{"role": "user", "content": f"Translate the following math word problem into {g.LNAME[lang]}. Keep every number exactly as written, in digits. Output only the translation.\n\n{ds[i]['question']}"}]) for i in idx], SamplingParams(temperature=0, max_tokens=700))
    texts = [o.outputs[0].text.strip() for o in tr]
    sol = llm.generate([chat([{"role": "system", "content": g.INSTR[lang]}, {"role": "user", "content": t}]) for t in texts], SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1536, seed=7))
    for i, t, o in zip(idx, texts, sol):
        gold = float(ds[i]["answer"].split("####")[-1].strip().replace(",", "")); txt = o.outputs[0].text.strip()
        print(json.dumps({"lang": lang, "gold": gold, "phrase_in": g.PHRASE[lang] in txt, "parsed": g.final_number(txt, lang), "finish": o.outputs[0].finish_reason, "tail": txt[-160:]}, ensure_ascii=False))
