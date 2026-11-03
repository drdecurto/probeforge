# Results

| Path | Contents |
|---|---|
| `reports/<eval-dir>/report*.json` | `contrastforge evaluate` reports (pass@1, worst language, per-view/family/language accuracy, root-bootstrap CI, tokens, length stops) for every evaluated arm. `cmp*.json` / `comparison*.json` are paired comparisons. Directory names follow the campaign: `evalv2` (Qwen3-8B), `eval14b` (Qwen3-14B incl. thinking/mixed targets), `evalphi4`, `evalphi4s` (Phi-4 seeds), `evalphi4mini`, `evalaya`, `evalv4`/`evalv4b`/`evalv4c` (mixed-format), `evalgrpo`, `q32_eval` (Qwen3-32B, seeds 1–3, `cmp_seeds.json`), `glm_eval` (GLM-4.7-Flash, seeds 1–3), `evalr4` (Phi-4 mixed-format seeds 1–3, Aya mined/mixed-format seeds 2–3), `kolibri_eval` (Kolibri-1 base, random and mixed-format seeds 1–3, `cmp_kol.json`), `evalmuse` (Muse-Spark-Open), `evalglm53` (GLM-5.3), `eval60*`/`evalpacks`/`evalthink*` (packs and thinking subsets). |
| `energybench/` | Summary CSVs (raw and chat protocols), key-value answer accuracy, analysis JSON, and `outputs_raw_and_chat.tgz` with every per-language output file (responses, per-problem scores, energy samples) for all evaluated checkpoints including Qwen3-32B and GLM-4.7-Flash; `outputs_post_v13.tgz` adds Qwen3-32B/GLM seeds 2–3, Phi-4 mixed-format, Aya seeds 2–3 and Kolibri-1. |
| `stdbench/` | `stdbench_summary.csv` (Belebele 12 languages, Global-MMLU-Lite 8, MGSM 6; strict and flexible extraction) and `lm_eval_results.tgz` with every lm-evaluation-harness `results_*.json`; `lm_eval_results_post_v13.tgz` adds the post-v13 runs, and `stdbench_kol.csv` gives Kolibri-1 (lm-eval `local-completions` backend). |
| `sig/` | Paired significance tables (main test, published benchmark, standard benchmarks) and per-model paired comparison files; `r4_energy_bootstrap.txt` and `kolibri_energy_bootstrap.txt` give per-seed paired energy intervals for the post-v13 runs. |

The complete per-prompt responses behind the paired analyses of the accompanying study are in `../paper/evidence/`.
