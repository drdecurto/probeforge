"""GRPO with the ProbeForge verifier as reward (TRL 0.18.2, HF generation, LoRA on a merged SFT policy).

usage: accelerate launch --num_processes 4 grpo_train.py --model MERGED_DIR --out RUN_DIR [--max-steps N] [--n-prompts 8000]
Reward = 1.0 if contrastforge.verify() accepts the response (status, answer, facts all exact), + 0.1 if it parses as the 4-key JSON.
Loss: Dr. GRPO (no std scaling, length-unbiased), beta = 0 (no reference model), truncated completions masked.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.expanduser("~/probeforge/src"))
from contrastforge.verify import verify, parse

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--train", default="/gpfs/scratch/bsc21/bsc021885/cf/data/contrast13m_200/train.jsonl")
ap.add_argument("--n-prompts", type=int, default=8000); ap.add_argument("--max-steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--num-generations", type=int, default=8)
ap.add_argument("--per-device-batch", type=int, default=8); ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--save-steps", type=int, default=25); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(a.train)]
rows = [r for r in rows if r.get("kind") == "math" and "world" in r]
random.Random(a.seed).shuffle(rows); rows = rows[:a.n_prompts]
render = lambda r: tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ds = Dataset.from_list([{"prompt": render(r), "row": json.dumps(r, ensure_ascii=False)} for r in rows])


def correct_reward(prompts, completions, row, **kw):
    return [1.0 if verify(json.loads(rw), c).get("correct") else 0.0 for c, rw in zip(completions, row)]


def format_reward(prompts, completions, **kw):
    out = []
    for c in completions:
        try:
            o = parse(c); out.append(0.1 if set(o) == {"status", "answer", "facts", "explanation"} else 0.0)
        except Exception:
            out.append(0.0)
    return out


cfg = GRPOConfig(
    output_dir=a.out, learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
    per_device_train_batch_size=a.per_device_batch, gradient_accumulation_steps=1,
    num_generations=a.num_generations, max_prompt_length=1024, max_completion_length=512, temperature=a.temperature,
    beta=0.0, loss_type="dr_grpo", scale_rewards=False, mask_truncated_completions=True, num_iterations=1,
    max_steps=a.max_steps, logging_steps=5, save_steps=a.save_steps, save_total_limit=3, bf16=True, seed=a.seed,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=False,
    report_to="none", model_init_kwargs={"torch_dtype": torch.bfloat16}, log_completions=False)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
trainer = GRPOTrainer(model=a.model, reward_funcs=[correct_reward, format_reward], args=cfg, train_dataset=ds,
                      processing_class=tok, peft_config=lora)
resume = any(d.startswith("checkpoint-") for d in os.listdir(a.out)) if os.path.isdir(a.out) else False
trainer.train(resume_from_checkpoint=True if resume else None)
trainer.save_model(a.out)
if trainer.is_world_process_zero():
    json.dump({"steps": trainer.state.global_step, "log_tail": trainer.state.log_history[-5:]}, open(f"{a.out}/metrics.json", "w"), indent=1)
