"""Real offline CPU training check: tiny random Qwen3, SFT, resume, DPO, reload, export.

This verifies software execution only. It is not an 8B capability experiment.
Run: python scripts/smoke_training.py --out /tmp/probeforge-smoke
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from safetensors.torch import load_file

from probeforge.tasks import generate_dataset
from probeforge.data import make_expert_data
from probeforge.io import read_jsonl, write_jsonl, write_json
from probeforge.train import train, cache_reference, export_model
from probeforge.policies import HFPolicy
from probeforge.environment import messages, public_state


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError("Use a fresh smoke-test directory")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    generate_dataset(root/"tasks", 24, 42)
    for split in ("train", "validation"):
        make_expert_data(root/f"tasks/{split}.jsonl", root/split)
        for name in ("sft", "preferences"):
            rows = list(read_jsonl(root/split/f"{name}.jsonl"))[:8]
            write_jsonl(root/split/f"{name}.jsonl", rows)
    template = "{% for m in messages %}{{m['role'] + ':\\n' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant:\\n'}}{% endif %}"
    texts = []
    for split in ("train", "validation"):
        for row in read_jsonl(root/split/"sft.jsonl"):
            texts.append(" ".join(m["content"] for m in row["prompt"])+row["completion"])
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=700, special_tokens=["<pad>", "<unk>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>", chat_template=template)
    base = root/"tiny-base"
    tokenizer.save_pretrained(base)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=2048, eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True))
    model.save_pretrained(base)
    cfg = {"model": str(base), "train_data": str(root/"train/sft.jsonl"),
           "validation_data": str(root/"validation/sft.jsonl"), "output_dir": str(root/"sft"),
           "max_length": 2048, "lora_rank": 4, "lora_alpha": 8, "learning_rate": .001,
           "batch_size": 1, "gradient_accumulation": 2, "allow_cpu": True,
           "gradient_checkpointing": True, "max_steps": 2, "eval_steps": 1, "save_steps": 1, "logging_steps": 1}
    initial = train(cfg)
    cfg["max_steps"] = 3
    resumed = train(cfg, resume="latest")
    weights = load_file(root/"sft/adapter_model.safetensors")
    assert any(v.abs().sum() > 0 for k, v in weights.items() if "lora_B" in k)
    dpo = {**cfg, "init_adapter": str(root/"sft"), "output_dir": str(root/"dpo"), "max_steps": 2,
           "learning_rate": .0001, "beta": .1, "train_data": str(root/"reference_train.jsonl"),
           "validation_data": str(root/"reference_validation.jsonl")}
    for split in ("train", "validation"):
        cache_reference(dpo, root/split/"preferences.jsonl", root/f"reference_{split}.jsonl")
    preferences = train(dpo, stage="dpo")
    assert abs(preferences["eval_loss"] - .693147) > 1e-6
    policy = HFPolicy(str(base), str(root/"dpo"), device="cpu", max_tokens=8)
    task = next(read_jsonl(root/"tasks/test_ood.jsonl"))
    text, usage = policy.generate(messages(public_state(task)))
    assert usage["completion_tokens"] > 0
    export_model(str(base), str(root/"dpo"), str(root/"merged"))
    assert (root/"merged/config.json").exists()
    report = {"passed": True, "kind": "tiny random Qwen3 CPU software smoke test",
              "sft": initial, "resumed_sft": resumed, "dpo": preferences,
              "adapter_reload_generated_tokens": usage["completion_tokens"], "merged_export": True,
              "not_tested": ["8B CUDA training", "4-bit CUDA", "MareNostrum Slurm", "live Nebius", "capability improvement"]}
    write_json(root/"smoke_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
