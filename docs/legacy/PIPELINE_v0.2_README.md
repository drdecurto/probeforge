# ProbeForge 0.2 — verified multilingual contrast training

An experimental pipeline for post-training Qwen3-8B with limited compute. It learns
to preserve answers across equivalent language/wording changes, respond to changed
facts and constraints, and identify missing assumptions, infeasibility and unboundedness.

**Start with the CPU demo, then a small GPU pilot.** This archive contains code and
instructions, not trained 8B weights. Read `VERIFICATION.md` for the executed checks.

## Included

- Twelve exact mathematical problem families in English, Spanish, Catalan, Arabic,
  and Luxembourgish; family-disjoint splits and grouped perturbations.
- Resumable local/Nebius-compatible collection with failures, finish reasons,
  token usage, and optional generation-only NVML energy.
- Failure-based group mining, translated/random baselines, a counterfactual ablation,
  and whole-group token-budget matching.
- SFT LoRA/QLoRA, optional cached-reference DPO, checkpoint resume and merged export.
- Per-language/view/family accuracy, all-views-correct, paired comparisons, root-group
  intervals and token/latency/energy metrics.
- LIC-inspired numerical policy training with changing optima and strong adaptive controls.
- Optional annotated-image count/spatial training with Qwen2.5-VL-3B, a frozen visual
  encoder, exact labels and reflection tests.
- Slurm scripts, experiment matrices, a Colab notebook and a Mac workflow.
- The original experiment-selection pipeline remains available under `probeforge`.

## 1. Free CPU demo — Mac or Linux

Python 3.11/3.12 is recommended. The core has no third-party dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
contrastforge demo --out demo_run
```

The demo generates data, collects **synthetic fixture responses**, mines examples
and writes a report. Its scores are not LLM results. Use a fresh output directory
to repeat a full demo.

## 2. GPU environment and model

On Colab A100 or compatible Linux CUDA:

```bash
python -m pip install -e '.[train,quant,energy]'
contrastforge doctor
contrastforge download --model Qwen/Qwen3-8B --out models/Qwen3-8B
```

Pins: Torch 2.6.0, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0. Install the
Torch build compatible with your site's driver/modules. Compute jobs do not
download models or install packages. Model downloads record the resolved revision.
Use the Mac for data and API orchestration; this training path requires CUDA.

## 3. Generate and inspect a small pilot

```bash
contrastforge generate --out data/contrast --groups 60
contrastforge preflight \
  --inputs data/contrast/train.jsonl data/contrast/validation.jsonl data/contrast/test.jsonl \
  --tokenizer models/Qwen3-8B --max-length 2048 --out data/preflight.json
```

Sixty underlying training worlds produce 1,600 training prompts, 300 validation
prompts and 300 test prompts. The full setting uses `--groups 1000` in a new directory.
Review the authored translations before making language-specific research claims.

## 4. Collect the student on training and validation views

```bash
contrastforge collect \
  --inputs data/contrast/train.jsonl data/contrast/validation.jsonl \
  --out data/student_base.jsonl --model models/Qwen3-8B \
  --quantize --budgets 512 --samples 1
```

The same command resumes if inputs/settings match. Recorded errors remain recorded;
retry them in a separate documented run. For candidate diversity use
`--samples 2 --temperature 0.6`. Do not mix different checkpoints in one mining run.
If no completely observed repairable groups exist, inspect truncation/failures,
increase the output budget, or design harder problems. `random` SFT remains useful
for an infrastructure check, but does not establish a mining improvement.

## 5. Prepare comparable training arms

```bash
for arm in translated random mined no_counterfactual; do
  contrastforge prepare \
    --inputs data/contrast/train.jsonl data/contrast/validation.jsonl \
    --rollouts data/student_base.jsonl --strategy "$arm" \
    --tokenizer models/Qwen3-8B --token-budget 2000000 \
    --out "data/curricula/$arm"
done
```

Manifests record actual sequence tokens, repetitions and unique examples. Whole
groups cause small budget differences. Train exactly one epoch to preserve the
comparison. Two million tokens is an initial experiment budget, not a cost estimate.

## 6. Train on Colab / one CUDA GPU

```bash
contrastforge train --config configs/contrast_sft_colab.json
```

This saves the adapter in `runs/contrast8b_colab_sft`. An infrastructure pilot can use:

```bash
contrastforge train --config configs/contrast_sft_colab.json \
  --max-steps 10 --output-dir runs/infrastructure_pilot
```

For an interrupted run with periodic checkpoints, add `--resume latest`.
`scripts/build_experiment_configs.py` prepares baseline/seed configs without launching jobs.

## 7. MareNostrum 5

Use project storage and a prepared environment on your accelerated allocation.

```bash
export CF_ACCOUNT='YOUR_PROJECT_ACCOUNT'
export CF_QOS='YOUR_ACCELERATED_QOS'
export CF_PYTHON="$PWD/.venv/bin/python"
# Optional: CF_PARTITION, CF_ENV_SETUP, CF_TIME, CF_GRES
bash slurm/submit_contrast.sh train --config configs/contrast_sft_mn5.json
```

The new launcher uses one process and requests one GPU by default. If your QoS
requires a whole node, set `CF_GPUS`/`CF_GRES` accordingly and count all allocated
GPU-hours. It does not automatically launch extra workers. The helper also accepts
`collect`, `cache-reference` and `train-vision`. See `docs/CONTRAST_RUNBOOK.md`.

## 8. Compare the original and adapted model

```bash
contrastforge collect --inputs data/contrast/test.jsonl \
  --out runs/base_test.jsonl --model models/Qwen3-8B --quantize --budgets 256 512 1024
contrastforge collect --inputs data/contrast/test.jsonl \
  --out runs/sft_test.jsonl --model models/Qwen3-8B --adapter runs/contrast8b_colab_sft \
  --quantize --budgets 256 512 1024
contrastforge evaluate --inputs data/contrast/test.jsonl \
  --rollouts runs/base_test.jsonl --out runs/base_report.json
contrastforge evaluate --inputs data/contrast/test.jsonl \
  --rollouts runs/sft_test.jsonl --out runs/sft_report.json
contrastforge compare --reports runs/base_report.json runs/sft_report.json --out runs/comparison.json
```

Use the MN5 adapter path if trained there. Add unrelated held-out tasks to assess
retention; this generated benchmark alone does not establish general improvement.

## Further workflows

- `docs/CONTRAST_RUNBOOK.md`: Nebius, DPO, LIC, VLM, Slurm matrices and troubleshooting.
- `docs/CONTRAST_METHOD.md`: objective, splitting, limitations and attribution.
- `notebooks/contrast_colab_a100.ipynb`: staged Colab workflow.
- `scripts/smoke_contrast_training.py`: real tiny-Qwen CPU training check.
- `scripts/smoke_vision_training.py`: optional tiny-VLM CPU execution check.
- `docs/LEGACY_README.md`: original experiment-selection pipeline.

The adapted text model runs without a teacher or ensemble at inference. Text and
vision adapters belong to different base models and must not be interchanged.
