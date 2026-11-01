"""Difficulty pre-pass for GRPO: sample k completions per training prompt with vLLM, verify, keep informative prompts.

usage: python prefilter.py --model MERGED_DIR --shard i --nshards n --pool 16000 --k 4 --out DIR
Writes DIR/pool_<i>.jsonl: the original row + "p_correct" (fraction of k samples verified correct).
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.expanduser("~/probeforge/src"))
from contrastforge.verify import verify

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="/gpfs/scratch/bsc21/bsc021885/cf/data/contrast13m_200/train.jsonl")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pool", type=int, default=16000); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(a.train)]
    rows = [r for r in rows if r.get("kind") == "math" and "world" in r]
    random.Random(a.seed).shuffle(rows); rows = rows[:a.pool][a.shard::a.nshards]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, seed=a.seed)
    tok = llm.get_tokenizer()
    ps = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    outs = llm.generate(ps, SamplingParams(n=a.k, temperature=a.temperature, top_p=1.0, max_tokens=512, seed=a.seed))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/pool_{a.shard}.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            c = sum(1 for x in o.outputs if x.finish_reason == "stop" and verify(r, x.text).get("correct"))
            r["p_correct"] = c / a.k; f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("shard", a.shard, "done", len(rows))


if __name__ == "__main__":
    main()
