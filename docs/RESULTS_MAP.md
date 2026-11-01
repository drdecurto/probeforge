# Results map for the accompanying study

Table and figure numbers follow the accompanying study. "Frozen" tables are typeset LaTeX kept under version control. Their
numbers come from the CSV snapshot in `paper/source_tables/`, which `paper/refresh_figures.py` regenerates
from the reports in `paper/data/`. Scripts are run from `paper/`.

## Tables

| # | Content | File | Source data | Generator |
|---|---|---|---|---|
| 1 | View definitions | (text of the study) | `src/contrastforge/worlds.py`, `languages.py` | – |
| 2 | Main benchmark, all arms | `tables/main.tex` | `paper/data/round4,6,8…`, `evidence/round4,6,8,19` | `refresh_v12.py` |
| 3 | Accuracy by language | `tables/languages.tex` | `source_tables/table2_per_language.csv` | frozen (`refresh_figures.py data` → CSV) |
| 4 | Accuracy by view (main and diag60) | `tables/views.tex` | `source_tables/table3_per_view.csv` | frozen |
| 5 | Paired six-view and all-language correctness | `tables/paired.tex` | `evidence/paired/`, `evidence/round20/{q32,glm}/eval/` | `analyze_paired.py` |
| 6 | core60 / diag60 packs | `tables/packs.tex` | `source_tables/table4_packs.csv` | frozen |
| 7 | development112 transfer | `tables/transfer.tex` | `source_tables/table6_dev112_transfer.csv` | frozen |
| 8 | Inference regimes (thinking / non-thinking) | `tables/regimes.tex` | `source_tables/table5_regimes.csv` | frozen |
| 9 | Unadapted references (Muse-Spark-Open, GLM-5.3) | `tables/frontier.tex` | `evidence/round14/report*_muse*.json`, `evidence/round20/evalglm53/report*.json` | frozen; checks: `analyze_glm53_diagnostics.py` |
| 10 | Published reasoning benchmark (S, K) | `tables/energybench.tex` | `source_tables/table7_energy_benchmark.csv`, `table9_answer_accuracy.csv`, `evidence/round15/` | frozen |
| 11 | Standard benchmarks, initial 24 configurations | `tables/standard.tex` | `source_tables/table8_standard_benchmarks.csv`, `evidence/round14,15/stdbench_summary.csv` | frozen |
| 12 | MGSM strict vs last-number extraction | `tables/extraction.tex` | same as 11 | frozen |
| 13 | Replicated transfer and cross-family limits | `tables/replicated.tex` | `evidence/round16–19/` | `refresh_v12.py` |
| 14 | Qwen3-32B and GLM-4.7-Flash extensions | `tables/scale.tex` | `evidence/round20/` | `analyze_round20.py` |

## Figures

| # | File | Generator |
|---|---|---|
| 1 | `figures/pipeline_diagram.pdf` | `figures/pipeline_diagram.tex` (`latexmk -pdf`) |
| 2 | `figures/fig1_arms_by_model.pdf` | `refresh_v12.py` |
| 3 | `figures/fig2_per_language.pdf` | `refresh_language_figure.py` (`evidence/language_plot_exact.json`) |
| 4 | `figures/fig3_per_view.pdf` | `refresh_figures.py data` |
| 5 | `figures/fig4_packs.pdf` | `refresh_figures.py data` |
| 6 | `figures/fig6_v3_seeds.pdf` | `refresh_figures.py data` |
| 7 | `figures/fig5_regimes.pdf` | `refresh_figures.py data` |
| 8 | `figures/fig7_energy_benchmark.pdf` | `refresh_v12.py` |
| 9 | `figures/fig8_energy_chat.pdf` | `refresh_v12.py` |
| 10 | `figures/fig9_standard_benchmarks.pdf` | `refresh_v8_figures.py` |
| 11 | `figures/fig10_disciplines.pdf` | `refresh_v8_figures.py` |

## Numbers quoted only in the text

| Quantity | Source | Script |
|---|---|---|
| Paired problem-bootstrap intervals for mixed-format transfer (Qwen3-8B/14B) | `evidence/transfer_intervals/` | `analyze_transfer_intervals.py` |
| Significance of main, energy and standard-benchmark changes | `evidence/table10_sig_{main,energy,standard}.csv` | `campaign/sig/sig.py` |
| 32B/GLM paired root-bootstrap gains (+0.306, +0.244) | `evidence/round20/{q32,glm}/eval/cmp.json` | `contrastforge compare` |
| GLM-4.7-Flash length stops (140 / 2 / 9) and GLM-5.3 failure breakdown (424 of 727) | `evidence/round20/` | `analyze_round20.py`, `analyze_glm53_diagnostics.py` |
| Counterfactual production-planning case | `evidence/counterfactual_case.json` | – |
| Mining group counts (60 + 15) | `evidence/paired/mining_diagnostics.jsonl`, `manifest.json` | `src/contrastforge/curriculum.py` |
| Campaign timings, resources and protocol diagnostics | `evidence/campaign_v08.md`, `campaign_sessions17_19.txt`, `round20/campaign_excerpt.txt` | – |
