---
language:
  - en
base_model: Qwen/Qwen3-8B
library_name: peft
tags:
  - lora
  - experimental-reasoning
  - tool-use
  - research
---

# Probe-8B — fill after training

**Status:** untrained template. Do not publish this as a trained model card until
the fields below are filled with actual artifacts and results.

## Model

- Exact base repository and revision: TO FILL.
- Adapter checksum and training configuration: TO FILL.
- SFT/DPO stage and seed: TO FILL.
- License and base attribution: retain the base model's applicable license and
  notices; document the chosen license for your adapter/data.
- Training software versions and hardware: TO FILL.
- Intended use: research on selecting experiments and using returned evidence in
  finite executable environments.

## Training data

- Dataset manifest checksums and task-family split: TO FILL.
- Number of unique groups, worlds, decision examples, preferences, and tokens: TO FILL.
- Teacher model/version and API-generated examples, if any: TO FILL.
- Reference model fingerprint for DPO: TO FILL.
- Filtering, deduplication, and data license: TO FILL.

## Evaluation

| Split | Budget | Base success | Adapted success | Paired interval | Tokens | Tool calls |
|---|---:|---:|---:|---|---:|---:|
| TO FILL | | | | | | |

Report optimization-seed variability, malformed-action failures, total compute,
and the exact evaluation manifest. Do not substitute exact-planner scores for model scores.

## Inference

Use the saved tokenizer chat template with thinking disabled and the documented
JSON action protocol. A runtime must execute probe actions and return observations.
The adapter does not contain an external simulator or test executor. Nebius is
unnecessary for inference after training.

## Limitations

The initial curriculum provides candidate formulas and a finite probe menu.
Evidence of success here is not proof of unrestricted experiment design, arbitrary
code repair, broad causal reasoning, or validated microgrid control. Realistic
domain transfer and retention of general language ability must be evaluated.

## Attribution

Name the base model, ProbeForge version, data authors, and methods used (LoRA/QLoRA,
DPO, active learning, teacher distillation). Describe the contribution as a derived
checkpoint and measured capability improvements rather than a new architecture.
