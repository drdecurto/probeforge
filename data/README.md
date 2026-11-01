# Data

| Path | Contents |
|---|---|
| `contrast13m_200/` | Main generated dataset (seed 42, 13 language configurations, metamorphic views): `train.jsonl` (21671 rows, 200 roots, families rate/affine/mean/variance/product/energy), `validation.jsonl` (4160, workrate/rectangle/system), `test.jsonl` (4160, journey/production/interest). `test_eval12.jsonl` is the 1248-prompt main test (12 roots × 13 × 8); `think_ids.json` is the 26-prompt thinking subset; `eval_ids.json`/`mining_ids.json` and `*shard*.json` are the id lists used for sharded collection. `manifest.json` records the dataset hash. |
| `packs/probeforge_eval_candidate_2026-10-13/` | core60 (`test_core60.jsonl`, 6240) and diag60 (`test_diagnostic60.jsonl`, 6240) held-out packs, `think60_ids.json` (52-prompt subset), answer keys, root catalogue, exclusion hashes, and the builder `build_and_validate.py` with a frozen reference copy of the generator. |
| `packs/probeforge_coverage_extension_2026-10-13/` | development112 (1824 prompts, 7 new families, EN/ES), native four-key variant, confirmation and pilot sets, builder and validation. |
| `energybench_langs/<lang>/problems.json` | The 79 problems × 13 languages of the published multilingual reasoning benchmark (de Zarzà et al., 2026). |
| `gsm8k_multilingual_verified/gsm_<lang>.jsonl` | 32 260 verified (problem, solution) pairs from the GSM8K *training* split in 13 languages, used by the mixed-format curricula. Translation by Qwen3-14B keeps every number; solutions are kept only when the final number equals the GSM8K reference. Derived from GSM8K (MIT). |
| `curricula/<name>/` | Manifests (token counts, rows, exposures, strategy, rollout run hash), `mix_stats.json` for mixed-format mixtures and gzipped mining diagnostics for every curriculum in the accompanying study. The full SFT files can be rebuilt with `contrastforge prepare` / `campaign/v4/v4_mix.py` from the files above. |
| `student_rollouts/*.jsonl.gz` | Verified base-model rollouts on the train/validation pool (Qwen3-8B `student_base_v2`, Qwen3-14B, Qwen3-14B status-diverse, Phi-4, Phi-4-mini, Aya-Expanse-8B), used for error mining. Decompress before `contrastforge prepare --rollouts`. |

All authored problem data are MIT-licensed; see `../NOTICE` for third-party sources.
