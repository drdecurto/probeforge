"""Completion-only labels, with strict length validation and no truncation."""
from .io import digest


def render_prompt(tokenizer, messages, thinking=False):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def encode_completion(tokenizer, messages, completion, max_length, thinking=False):
    """`thinking=True` renders the prompt under the model's reasoning template (Qwen3: no empty <think/> block
    is inserted), so the completion is expected to carry its own <think>...</think> prefix (mixed-mode SFT)."""
    prompt = tokenizer.encode(render_prompt(tokenizer, messages, thinking=thinking), add_special_tokens=False)
    answer = tokenizer.encode(completion + tokenizer.eos_token, add_special_tokens=False)
    if not answer:
        raise ValueError("Empty completion")
    if len(prompt) + len(answer) > max_length:
        raise ValueError(f"Example has {len(prompt)+len(answer)} tokens > max_length={max_length}; raise max_length or filter explicitly")
    return {"input_ids": prompt+answer, "attention_mask": [1]*(len(prompt)+len(answer)),
            "labels": [-100]*len(prompt)+answer}


def encode_row(tokenizer, row, max_length, stage="sft"):
    thinking = bool(row.get("thinking", False))
    if stage == "sft":
        return encode_completion(tokenizer, row["prompt"], row["completion"], max_length, thinking)
    result = {}
    for side in ("chosen", "rejected"):
        encoded = encode_completion(tokenizer, row["prompt"], row[side], max_length, thinking)
        result.update({f"{side}_{key}": value for key, value in encoded.items()})
        if f"ref_{side}" in row:
            result[f"ref_{side}"] = row[f"ref_{side}"]
    return result


class CompletionCollator:
    def __init__(self, pad_id, stage="sft"):
        self.pad_id, self.stage = pad_id, stage

    def __call__(self, rows):
        import torch
        batch = {}
        prefixes = [""] if self.stage == "sft" else ["chosen_", "rejected_"]
        for prefix in prefixes:
            size = max(len(r[prefix+"input_ids"]) for r in rows)
            for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
                name = prefix + key
                batch[name] = torch.tensor([r[name]+[pad]*(size-len(r[name])) for r in rows], dtype=torch.long)
        if self.stage != "sft":
            for side in ("chosen", "rejected"):
                name = f"ref_{side}"
                if name in rows[0]:
                    batch[name] = torch.tensor([r[name] for r in rows], dtype=torch.float32)
        return batch


def sequence_logps(logits, labels):
    """Sum log p of completion tokens only; chunk FP32 conversion to bound memory."""
    import torch
    from torch.nn import functional as F
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    total = torch.zeros(labels.shape[0], device=logits.device, dtype=torch.float32)
    for start in range(0, labels.shape[1], 64):
        y = labels[:, start:start+64]
        x = logits[:, start:start+64, :].float()
        loss = F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none")
        total = total - loss.reshape(y.shape).sum(dim=1)
    return total


def row_fingerprint(row):
    return digest({k: row[k] for k in ("prompt", "chosen", "rejected")})
