# Reproducing the experiments

Paths below are relative to the repository root. The campaign scripts in `campaign/` are the exact files run
on MareNostrum 5; they contain that cluster's absolute paths (`/gpfs/scratch/...`, variable `S`). Adapt `S`,
the Slurm account/QoS and environment activation before use.

## 1. Data

```bash
contrastforge generate --out data/contrast13m_200 --groups 200 --metamorphic      # seed 42, 13 languages
```

reproduces `data/contrast13m_200/{train,validation,test}.jsonl` (21671 / 4160 / 4160 rows; family-disjoint
splits; dataset hash in `manifest.json`). The main test `test_eval12.jsonl` (1248 prompts) is the 12-root
subset listed in `eval_ids.json`; `think_ids.json` lists the 26-prompt thinking subset. The additional packs
(`data/packs/probeforge_eval_candidate_2026-10-13`: core60, diag60, think60 ids;
`.../probeforge_coverage_extension_2026-10-13`: development112) include their own
`build_and_validate.py` and answer keys.

## 2. Base-model rollouts and curricula

```bash
contrastforge collect --inputs data/contrast13m_200/train.jsonl data/contrast13m_200/validation.jsonl \
  --out data/student_base.jsonl --model models/Qwen3-8B --budgets 512 --samples 1
contrastforge prepare --inputs data/contrast13m_200/train.jsonl data/contrast13m_200/validation.jsonl \
  --rollouts data/student_base.jsonl --strategy {translated|random|mined|no_counterfactual} --targets worked \
  --tokenizer models/Qwen3-8B --token-budget 3000000 --out data/curricula/<arm>
```

The archived mining rollouts are in `data/student_rollouts/` (gzipped) and every curriculum's manifest and
mining diagnostics in `data/curricula/`. Mining selects complete groups by
`2·1{successes and failures} + b_g/m_g`, keeps all positive-priority groups plus round(H/4) background
groups, and packs whole groups into the budget (`src/contrastforge/curriculum.py`).

Mixed-format curricula (v4, v4b, v4c): `campaign/v4/gsm_ml.py` translates and verifies GSM8K training
problems (pool in `data/gsm8k_multilingual_verified/`); `campaign/v4/v4_mix.py` mixes 50 % contrast JSON,
25 % prose and 25 % GSM8K by tokens of the target model. Seeds 1–3 vary the mixture sampling.

## 3. Training

```bash
contrastforge train --config configs/v4c_8b_s1.json [--resume latest]
```

One GPU per arm (bf16 LoRA r16/α32, micro-batch 1 × 32 accumulation, lr 1e-4, 1 epoch); Qwen3-32B uses
4-bit QLoRA (`v32b_*`), GLM-4.7-Flash 2 GPUs per arm (`vglm_*`), Kolibri-1 four GPUs per arm (`vkol_*`). Seeds 2–3 use the `_s2`/`_s3`
configs. Slurm chains in `campaign/chains/` (e.g. `q32_chain.sh`, `glm_chain.sh`, `master_1101.sh`,
`kolsft_chain.sh`) resubmit 2-hour jobs until `metrics.json` exists. Kolibri-1 adapters are merged with
`campaign/kolibri/merge_lora.py` and evaluated by `campaign/kolibri/serve_eval.sh` (`kolpost_chain.sh`).

## 4. Evaluation on ProbeForge

```bash
contrastforge collect --inputs data/contrast13m_200/test_eval12.jsonl --out runs/test_<arm>.jsonl \
  --model models/Qwen3-8B --adapter runs/<arm> --budgets 512 --samples 1
contrastforge evaluate --inputs data/contrast13m_200/test_eval12.jsonl --rollouts runs/test_<arm>.jsonl --out report.json
contrastforge compare --reports report_base.json report_<arm>.json --out cmp.json   # paired root bootstrap
```

For served models (`--backend api`), pass `--api-extra campaign/q32/api_extra.json` to disable thinking and
shard prompts with `--only-ids` (`campaign/q32/remaining.py`).

## 5. External benchmarks

* Published multilingual reasoning benchmark: `campaign/energybench/eval_patched.py` (raw completion) and
  `eval_chat.py` (chat template), problems in `data/energybench_langs/`; key-value match with
  `campaign/energybench/answer_acc.py`.
* Belebele / Global-MMLU-Lite / MGSM: `campaign/stdbench/node_std.sh` (task lists `tasklist.txt`, models
  `models.txt`), aggregated by `campaign/stdbench/aggregate.py` (Kolibri-1: `aggregate_env.py` with
  `MODELS=models_kol.txt`, results from the `local-completions` backend).
* Significance tables: `campaign/sig/sig.py`; per-seed paired energy intervals: `campaign/sig/r4_energy.py`,
  `campaign/sig/kol_energy_boot.py` (outputs in `results/sig/*_bootstrap.txt`).

## 6. Tables and figures

See `paper/README.md` and `docs/RESULTS_MAP.md`.
