# ProbeForge — train your own model to investigate before answering

ProbeForge is a complete research pipeline for creating a **derived Qwen model**
that selects useful tests, incorporates their actual results, and stops within a
limited experimentation budget. The working checkpoint name is **Probe-8B**.
This repository contains training code, not a pretrained Probe-8B checkpoint.

Start with Qwen3-8B. The pinned stack also supports Qwen2.5 models. A model trained
with this pipeline is a new derivative checkpoint with its own data and evaluation;
it is not a newly pretrained foundation model or a new transformer architecture.

## What is implemented

- Deterministic, paired environments with eight possible hidden worlds per group.
- Twelve executable numerical, sequence-processing, and toy battery rule families.
- Six families for training and six structurally held-out families for evaluation.
- Exact dynamic-programming teacher using **only publicly available information**.
- SFT data and preference data without any paid API calls.
- Local Hugging Face and Nebius/OpenAI-compatible rollout backends.
- Verified teacher demonstrations, student collection, and counterfactual branching.
- Completion-only LoRA/QLoRA SFT; DPO with cached fixed-reference probabilities.
- Checkpoint save/resume, adapter loading, and merged checkpoint export.
- Equal-budget evaluation, invalid-action accounting, group-bootstrap intervals,
  paired comparisons, JSON/CSV reports, and individual trajectories.
- Single-node multi-GPU Slurm launcher, Colab notebook, and offline smoke tests.

**Scope:** v0.1 gives the model the candidate formulas and a finite test menu. It
learns to investigate within that hypothesis space and predict three target values.
It does not yet generate arbitrary programs, design unrestricted experiments, or
control a real microgrid. The battery tasks use simplified integer rules, not
validated electrical models. See [docs/METHOD.md](docs/METHOD.md).

## Where to run each part

| Machine | Recommended role | Configuration |
|---|---|---|
| Mac M3, 32 GB | Generate/audit datasets; orchestrate Nebius; inspect reports; small-model MPS inference | Standard-library core; optional training dependencies for inference |
| Colab A100 | First 8B training pilot; optionally DPO | `configs/sft_colab.json`, 4-bit QLoRA |
| MareNostrum 5 accelerated node | Main runs and multiple seeds | `configs/sft_mn5.json`, BF16 LoRA, four local GPU processes |

Use **one MN5 node initially**. One training process per GPU holds its own frozen
8B backbone; this is data parallelism, not four GPUs acting as one pooled memory.
The 4-GPU config and Colab config both have an effective batch of 32 examples.
Quantization is an additional experimental difference; do not call the two runs
strict replications. Start with 300 groups as a smoke/pilot dataset, then 1,500–3,000
groups in a fresh dataset directory for a substantive first experiment.

## 1. Install and verify locally

Python 3.11 or 3.12 is recommended. Work from the extracted repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
probeforge doctor
bash scripts/prepare_data.sh
```

The core has **no external dependencies**. Torch-dependent tests are skipped when
Torch is absent. `prepare_data.sh` creates a fresh dataset, verifies checksums and
group splits, builds train/validation examples, and evaluates symbolic baselines.
It refuses to overwrite an existing dataset. For a larger fresh run:

```bash
probeforge generate --out data/tasks_large --groups 1500 --seed 42
probeforge audit --data data/tasks_large
probeforge build --tasks data/tasks_large/train.jsonl --out data/large_train
probeforge build --tasks data/tasks_large/validation.jsonl --out data/large_validation
```

Use the corresponding paths as training overrides. Keep each dataset manifest.
Never put `test_id.jsonl` or `test_ood.jsonl` into training or preference collection.

## 2. Prepare the model and GPU environment

On the Linux training machine, create its own environment (do not copy a Mac venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
# Colab/QLoRA only:
python -m pip install -e '.[quant]'
probeforge doctor
probeforge download --model Qwen/Qwen3-8B --out models/Qwen3-8B
probeforge preflight --config configs/sft_mn5.json
```

Download on an internet-enabled machine or permitted transfer/login service,
**before** starting offline compute jobs. `download` resolves and records an exact
Hub commit. Transfer the entire `models/Qwen3-8B` folder to MN5 if needed. It is
roughly 16–17 GB of weights, plus small tokenizer/configuration files. Keep ample
space for the environment, checkpoints, data, and an optional merged export.

The pinned core versions are Torch 2.6.0, Transformers 4.51.3, PEFT 0.15.2, and
Accelerate 1.6.0. This is a deliberate Qwen3-compatible baseline, not a claim that
these are the newest releases. Newer Qwen3.5/GLM architectures require a separately
validated dependency stack. See [docs/HARDWARE.md](docs/HARDWARE.md).

## 3. Run the first training pilot

On one A100, after preparing data and downloading the model:

```bash
probeforge train --config configs/sft_colab.json --max-steps 30
```

This checks memory, throughput, saving, and loss. Resume into a longer run using:

```bash
probeforge train --config configs/sft_colab.json --max-steps 200 --resume latest
```

For a clean full run, use a fresh output directory and omit `--max-steps`:

```bash
probeforge train --config configs/sft_colab.json --output-dir runs/probe8b_colab_full
```

Colab workflow: open [notebooks/colab_a100.ipynb](notebooks/colab_a100.ipynb).
It saves checkpoints to Google Drive and separates setup, training, and evaluation.

On MareNostrum, set your actual allocated account and accelerated QoS:

```bash
export PF_ACCOUNT='YOUR_BSC_PROJECT'
export PF_QOS='YOUR_ACCELERATED_QOS'
export PF_PYTHON="$PWD/.venv/bin/python"
# Only if your allocation requires an explicit partition:
# export PF_PARTITION='YOUR_ACCELERATED_PARTITION'

bash slurm/submit.sh sft configs/sft_mn5.json --max-steps 30 --output-dir runs/mn5_pilot
```

Review `logs/`, `runs/mn5_pilot/metrics.json`, and `run_manifest.json`. Then:

```bash
bash slurm/submit.sh sft configs/sft_mn5.json
```

No job is submitted by extracting or installing this repository. Your account,
QoS, Python/modules, and partition are deliberately not guessed. See the complete
[Slurm instructions](docs/MARENOSTRUM.md), including reference caching and evaluation.

## 4. Evaluate the base and trained model with identical tools

Start with a few **validation** groups while choosing settings. Freeze settings
before running the held-out tests. The commands below are final-test examples:

```bash
probeforge evaluate --tasks data/tasks/test_ood.jsonl --backend hf \
  --model models/Qwen3-8B --out runs/base_ood --budgets 0 1 2 4 --max-groups 6

probeforge evaluate --tasks data/tasks/test_ood.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --out runs/sft_ood --budgets 0 1 2 4 --max-groups 6

probeforge compare --a runs/base_ood/episodes.jsonl \
  --b runs/sft_ood/episodes.jsonl --out runs/sft_vs_base.json
```

On Colab, append `--quantize` to **both** evaluation commands and use its adapter
path. Repeat on `test_id.jsonl`. Remove `--max-groups` for the full evaluation.
Keep decoding settings, model revision, seeds, tool budgets, and task sets equal.
The runtime enforces the budget, rejects invalid actions, and supplies actual probe
outputs. There is no repair of malformed JSON hidden inside the evaluation.

The symbolic `oracle`, `no_probe`, and `random` backends are **algorithmic bounds
or controls**, not measured LLM performance. They compute candidate predictions
exactly. The HF base with `--budgets 0` is the actual no-tool LLM baseline.

## 5. Add Nebius demonstrations (optional)

The first training run does not need Nebius. When useful, add verified trajectories
from a stronger model using the same public information and the same tool budget:

```bash
# Set the key securely in your shell; do not commit it or put it in a notebook.
export NEBIUS_API_KEY='YOUR_KEY'
probeforge models
export PF_TEACHER='COPY_AN_AVAILABLE_MODEL_ID'

probeforge collect --tasks data/tasks/train.jsonl --backend api \
  --model "$PF_TEACHER" --source nebius_teacher --out data/nebius_teacher \
  --limit 100 --max-requests 400 --max-tokens 256

probeforge mix --inputs data/expert_train/sft.jsonl \
  data/nebius_teacher/sft_successes.jsonl --out data/mixed_train.jsonl

probeforge train --config configs/sft_mn5.json --train-data data/mixed_train.jsonl \
  --output-dir runs/probe8b_teacher_mix
```

The default API URL is `https://api.tokenfactory.nebius.com/v1`. Override
`NEBIUS_BASE_URL` if your account uses another endpoint. Use a model supporting
concise JSON output. If it spends its token budget thinking, select an appropriate
non-thinking model or use its documented provider-specific option via `--extra-json`.
Do not silently increase the budget and then compare against a cheaper baseline.

Calls are cached and completed episodes are flushed to disk. Resume with the same
command plus `--resume`; `--limit` counts **new episodes** on that invocation.
`--max-requests` caps HTTP attempts, including retries, per invocation. Token usage
is logged from the provider; it is not a dollar spending cap. Cached calls do not
spend HTTP attempts. Prompt/response caches contain your task data, never the key.

## 6. Collect the student's own attempts and compare alternate actions

This is the optional second stage that addresses the student's current weaknesses.
Use the **same student checkpoint** for collection and continuation rollouts:

```bash
probeforge collect --tasks data/tasks/train.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --out data/student_round1 --limit 200

probeforge branch --tasks data/tasks/train.jsonl \
  --episodes data/student_round1/episodes.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --out data/branches_round1 --max-states 100 --worlds 4 --repeats 2 --max-actions 4

probeforge mix --inputs data/expert_train/preferences.jsonl \
  data/branches_round1/preferences.jsonl --out data/preferences_round1.jsonl
```

At every selected state, branching compares a public-information stopping answer,
the student's action when valid, and randomly sampled available probes. It holds
the sampled hidden world and continuation seed constant across alternatives.
It scores actual final success minus probe cost, averaging over plausible worlds.
No hidden truth enters the continuation prompt. Margins are noisy Monte Carlo
estimates; inspect `audit.jsonl`. A deterministic model yields duplicate repeats,
so use `--repeats 1` at temperature zero, or explicitly choose a sampling temperature.

Upper bound at these settings: 100 states × 4 alternatives × 4 worlds × 2 repeats
= 3,200 continuation episodes, each bounded by the runtime. Begin with 10 states.
The cheap exact-planner preference data can be used without this stage.

## 7. Preference training (DPO)

Choose a preference source. Cache reference probabilities using the **unchanged SFT
adapter** and the same precision/configuration as DPO. This avoids holding a second
8B model in GPU memory during training.

```bash
probeforge cache-reference --config configs/dpo_mn5.json \
  --input data/preferences_round1.jsonl --out data/reference_train.jsonl

probeforge cache-reference --config configs/dpo_mn5.json \
  --input data/expert_validation/preferences.jsonl --out data/reference_validation.jsonl

probeforge train --config configs/dpo_mn5.json --stage dpo
```

For the cheaper experiment, replace the first input with
`data/expert_train/preferences.jsonl`. On Colab use `configs/dpo_colab.json` for
**both** caching and training. On MN5 use the reference and DPO Slurm commands in
the dedicated guide. The cache records the exact base/adapter fingerprint, row
content, and token limit; mismatched caches are rejected.

Evaluate SFT and DPO separately. Keep SFT if DPO reduces verified success or spends
more computation without a useful gain. For another round, collect from the new
checkpoint, write new data/output paths, and update `init_adapter` consistently.

## 8. Export and use your model

The trained adapter and tokenizer are already saved in `runs/...`. To run them,
use `--model models/Qwen3-8B --adapter runs/probe8b_dpo_seed42` with the HF backend.
To create a standalone merged Hugging Face checkpoint:

```bash
probeforge export --model models/Qwen3-8B --adapter runs/probe8b_dpo_seed42 \
  --out models/Probe-8B-v0.1
```

Merge on a machine with sufficient host RAM; a full unquantized 8B model must be
loaded. Keep the original adapter and base provenance. The merged model can be
used as a standard causal LM, but its experimental capability still requires a
runtime that implements the JSON probe protocol. Nebius is not needed at inference.
Use [MODEL_CARD_TEMPLATE.md](MODEL_CARD_TEMPLATE.md) when documenting a trained release.

## Verification and evidence

```bash
bash scripts/check.sh
python scripts/smoke_training.py --out /tmp/probeforge-smoke-new
```

The second command needs `.[train]` and trains a tiny randomly initialized Qwen3
offline on CPU. It checks SFT updates, checkpoint resume, DPO, adapter reload, and
merged export. It does not measure 8B quality. See [VERIFICATION.md](VERIFICATION.md)
for what was actually executed while preparing this release.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the ablation plan, stopping gates,
and compute accounting. No performance gains are claimed before those experiments.
