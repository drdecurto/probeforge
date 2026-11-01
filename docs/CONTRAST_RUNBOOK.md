# Execution runbook

## Mac M3 32 GB

Use the dependency-free core for generation, exact verification, inspection, LIC
search and API orchestration. Install `.[vision]` for image preparation. `allow_cpu`
is for tiny software checks; full training uses CUDA. Optional MLX/llama.cpp
conversion of exported text weights is outside this release and has not been tested.

## Nebius teacher assistance

Copy the current model identifier and base URL from your Nebius console. Include
`/v1` if required, without `/chat/completions`. Keys are environment variables and
are never written to manifests. Do not place keys in notebooks or scripts.

```bash
read -s -p 'Nebius API key: ' NEBIUS_API_KEY
export NEBIUS_API_KEY
export NEBIUS_BASE_URL='COPY_CURRENT_BASE_URL_FROM_CONSOLE'
export NEBIUS_TEACHER='COPY_CURRENT_MODEL_IDENTIFIER'
contrastforge collect \
  --inputs data/contrast/train.jsonl data/contrast/validation.jsonl \
  --only-ids data/curricula/mined/repair_ids.json \
  --backend api --base-url "$NEBIUS_BASE_URL" --model "$NEBIUS_TEACHER" \
  --out data/teacher_repairs.jsonl --budgets 1024
contrastforge accept-teacher \
  --inputs data/contrast/train.jsonl data/contrast/validation.jsonl \
  --rollouts data/teacher_repairs.jsonl --out data/teacher_sft_review.jsonl
```

Only independently correct complete responses are accepted. Review language and
explanations before adding them to SFT, preserve split labels and rematch the token
budget. Teacher assistance is optional: mathematical targets are generated locally.

For provider-specific decoding, pass `--api-extra file.json`, for example a file
containing `{"chat_template_kwargs":{"enable_thinking":false}}`, only if the endpoint
supports it. Other allowed fields are `reasoning_effort` and `top_p`. Local Qwen3 uses
`enable_thinking=False`; API defaults may differ. Match settings before comparing.
Unsupported parameters are logged as errors; seeds do not guarantee server determinism.

Sources: https://docs.tokenfactory.nebius.com/quickstart and
https://docs.tokenfactory.nebius.com/api-reference/introduction

## Optional DPO

Complete SFT first. Both preference split files must be nonempty. If suitable
complete student errors are absent, collect harder examples or omit DPO. Do not
fabricate validation preferences. Error/length-stopped responses are excluded.

```bash
contrastforge cache-reference --config configs/contrast_dpo_colab.json \
  --input data/curricula/mined/preferences_train.jsonl --output data/reference_train.jsonl
contrastforge cache-reference --config configs/contrast_dpo_colab.json \
  --input data/curricula/mined/preferences_validation.jsonl --output data/reference_validation.jsonl
contrastforge train --config configs/contrast_dpo_colab.json --stage dpo
contrastforge export --model models/Qwen3-8B --adapter runs/contrast8b_colab_dpo --out models/contrast8b_merged
```

Use the corresponding MN5 config/paths on MareNostrum. Cached log probabilities
use the frozen SFT reference, avoiding two live GPU backbones during DPO. Changes
to weights, quantization, tokenizer, examples or max length invalidate the cache.
Merged export uses unquantized BF16 weights and requires enough host RAM and disk.

## LIC-inspired curriculum

```bash
contrastforge lic-generate --out data/lic --groups 60
contrastforge lic-baselines --input data/lic/test.jsonl --out data/lic_baselines.jsonl
contrastforge collect --inputs data/lic/train.jsonl data/lic/validation.jsonl \
  --model models/Qwen3-8B --quantize --out data/lic_student.jsonl --budgets 512
contrastforge prepare --inputs data/lic/train.jsonl data/lic/validation.jsonl \
  --rollouts data/lic_student.jsonl --strategy mined --tokenizer models/Qwen3-8B \
  --token-budget 1000000 --out data/lic_curriculum
contrastforge train --config configs/contrast_sft_colab.json \
  --train-data data/lic_curriculum/sft_train.jsonl \
  --validation-data data/lic_curriculum/sft_validation.jsonl --output-dir runs/lic_adapter
```

The state contains a small graph, benefits, costs, fatigue and budget. The model
selects nodes and intensity. Exact bounded search supplies an optimum, and all
optimal actions are accepted. Counterfactual benefits/fatigue force a different
best action. Rewiring requires current-state reasoning but need not change an optimum.

Baselines are no intervention, frozen degree ranking, adaptive degree ranking,
adaptive greedy using the same objective, and exhaustive optimum. Report invalid
policy frequency alongside regret. Graph families are held out. English formal
objective text is shared across languages; this is not a fully translated benchmark.

This bounded curriculum is **not** a reproduction of LIC's diffusion dynamics,
confidence-as-dose parameter, themes or human social effects. Utilities are
stipulated. Train a separate adapter before testing a mixed math/policy curriculum.
To mix, concatenate train files together and validation files together, then
prepare the mixture with a fresh token budget and output directory.

## Optional vision-language branch

This branch uses `Qwen/Qwen2.5-VL-3B-Instruct`, freezes its visual encoder and trains
text-attention LoRA. It consumes raw vision rows so image information is retained.
The text `prepare` command rejects vision rows.

Provide one annotation record per image:

```json
{"image":"frame001.jpg","scene_id":"drive_001","split":"train","annotations_complete":true,"vocabulary":["car","pedestrian"],"objects":[{"category":"car","bbox":[0.1,0.2,0.3,0.7]}]}
```

Boxes are normalized `[xmin,ymin,xmax,ymax]`. Include all three splits with distinct
scenes/images. `annotations_complete=true` asserts exhaustive annotations for the
listed categories; do not use it for sparse labels. Obtain images/annotations from
their dataset's access route. No BDD100K images or unpublished annotations ship here.

```bash
python -m pip install -e '.[train,quant,vision]'
contrastforge download --model Qwen/Qwen2.5-VL-3B-Instruct --out models/Qwen2.5-VL-3B-Instruct
contrastforge vision-prepare --manifest annotations.jsonl --image-root /absolute/path/to/images --out data/vision
contrastforge train-vision --config configs/contrast_vlm_colab.json
contrastforge collect --inputs data/vision/test.jsonl --vision \
  --model models/Qwen2.5-VL-3B-Instruct --adapter runs/contrast_vlm_colab \
  --quantize --out runs/vision_test.jsonl --budgets 512
contrastforge evaluate --inputs data/vision/test.jsonl --rollouts runs/vision_test.jsonl --out runs/vision_report.json
```

Repeat collection without the adapter for a baseline. Counts include zeros, so
hallucinated categories are penalized. Left/right/center targets transform under
reflection; total counts remain invariant. Image hashes catch stale files. Re-run
preparation after moving images to another machine because paths are absolute.

This evaluates counts and horizontal spatial grounding, not driving safety,
bounding-box localization accuracy or general scene descriptions. Training uses
one image per batch, caps visual tokens through `max_pixels`, and rejects excessive
sequence lengths instead of truncating image tokens.

## MareNostrum execution

Prepare Python 3.11/3.12 with site-compatible CUDA modules. Use accelerated nodes
and project storage; do not train on a login node. `CF_ENV_SETUP` may point to a
shell file loading your authorized modules. Site/project identifiers are not guessed.

```bash
export CF_ACCOUNT='YOUR_ACCOUNT'
export CF_QOS='YOUR_ACCELERATED_QOS'
export CF_PYTHON="$PWD/.venv/bin/python"
export CF_TIME=04:00:00
# export CF_ENV_SETUP="$PWD/slurm/environment.local.sh"
# export CF_PARTITION='YOUR_PARTITION'
# export CF_GPUS=4                 # when whole-node allocation is required
# export CF_GRES='gpu:4'          # follow your allocation's resource syntax
bash slurm/submit_contrast.sh collect \
  --inputs data/contrast/train.jsonl data/contrast/validation.jsonl \
  --out data/student_base.jsonl --model models/Qwen3-8B --budgets 512
bash slurm/submit_contrast.sh train --config configs/contrast_sft_mn5.json
```

Wait for collection and prepare the curriculum before training. The helper returns
a job ID. Set `CF_DEPENDENCY=JOB_ID` for an `afterok` dependency; join multiple IDs
with `:` and unset the variable when finished. Inspect `squeue` and `sacct` output.
Single-process jobs may reserve unused GPUs if whole nodes are required; report
all allocated GPU-hours. Extra workers are not launched automatically.

After a promising pilot, create the default 12-run matrix:

```bash
python scripts/build_experiment_configs.py --base configs/contrast_sft_mn5.json \
  --curricula-root data/curricula --out configs/matrix
export CF_MATRIX_DIR="$PWD/configs/matrix"
export CF_ARRAY='0-11%2'
bash slurm/submit_contrast.sh matrix
unset CF_ARRAY
```

Four arms × three seeds; `%2` caps concurrency. A smaller matrix uses
`--arms translated mined --seeds 42` and `CF_ARRAY='0-1%1'`. Inspect token spread.
Each run has its own output directory. One epoch preserves the matched token budget.

For energy use `collect --nvml-indices PHYSICAL_INDEX` on an exclusive GPU.
CUDA/NVML UUIDs must match; never assume both indices are zero. Omit the option
on unsupported or MIG setups. Compute jobs default to offline Hugging Face access.

References: https://slurm.schedmd.com/gres.html and https://arxiv.org/abs/2503.09917

## External retention tasks

Use reviewed held-out tasks with an explicit JSON response contract. This adapter
does not download a benchmark or certify open-ended prose. Manifest example:

```json
{"source":"MY_DATASET@REVISION","group_id":"item_001","task":"multiple_choice","language":"en","prompt":[{"role":"user","content":"YOUR QUESTION AND OPTIONS. Return only JSON with answer equal to the option letter."}],"expected_response":{"answer":"B"}}
```

```bash
contrastforge import-retention --manifest retention_manifest.jsonl --out data/retention_test.jsonl
contrastforge collect --inputs data/retention_test.jsonl --model models/Qwen3-8B --quantize --out runs/retention_base.jsonl
contrastforge collect --inputs data/retention_test.jsonl --model models/Qwen3-8B --adapter runs/contrast8b_colab_sft --quantize --out runs/retention_sft.jsonl
contrastforge evaluate --inputs data/retention_test.jsonl --rollouts runs/retention_base.jsonl --out runs/retention_base_report.json
contrastforge evaluate --inputs data/retention_test.jsonl --rollouts runs/retention_sft.jsonl --out runs/retention_sft_report.json
contrastforge compare --reports runs/retention_base_report.json runs/retention_sft_report.json --out runs/retention_comparison.json
```

Imported retention tasks are permanently assigned to test and rejected by training
preparation. Keep all variants under one source/group identity. Grade multiple-choice
or other exact structured outputs; use the original benchmark evaluator separately
for tasks requiring specialized or open-ended grading.

## Reproducibility and troubleshooting

```bash
python scripts/fetch_evaluation_sources.py --out sources
contrastforge import-legacy --paths sources/crossplatform-llm sources/comparison_llm_decurto \
  --out data/published_diagnostics.jsonl
python -m unittest discover -s tests -v
python scripts/smoke_contrast_training.py --out /tmp/contrast_smoke_fresh
```

Source downloads are pinned and optional. No downloaded scripts/notebooks execute.
Imported references remain unverified diagnostics. Preserve dataset manifests,
raw responses, failure/usage logs, checkpoint provenance and run configurations.

- Overlength: explicitly increase context or reduce input/visual size.
- No mined groups: inspect missing/truncated responses and saturation; collect
  complete views or design harder tasks. Random SFT is an infrastructure fallback.
- Empty DPO validation: collect suitable errors or omit DPO.
- Resume mismatch: use a fresh collection file after data/decoding changes.
  If a crash leaves a partial final JSONL line, preserve the broken file and repair
  that line explicitly; malformed records are not silently skipped.
- CUDA OOM: use QLoRA, batch size one, shorter context and gradient accumulation.
- Missing energy: expected on APIs, CPU/Mac or unsupported NVML; retain null values.
- Language fidelity: exact semantic checks do not certify translations or prose.
