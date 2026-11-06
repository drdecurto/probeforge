# Reproduction package for the accompanying study

Everything here runs on a CPU in a few minutes (`pip install numpy matplotlib reportlab`; Python ≥ 3.10).

```bash
cd paper
python refresh_figures.py data        # rebuilds source_tables/*.csv and Figures 4–7 from paper/data reports
python refresh_language_figure.py     # Figure 3
python refresh_v8_figures.py          # Figures 10–11
python refresh_v12.py                 # Tables 2 and 13, Figures 2, 8, 9 (rounds 16–19 evidence)
python analyze_paired.py              # Table 5 (recomputed from raw responses; checks hashes and archived pass@1)
python analyze_round20.py             # Table 14 + VALIDATION_ROUND20.json (Qwen3-32B, GLM-4.7-Flash)
python analyze_transfer_intervals.py  # paired bootstrap intervals for mixed-format transfer
python analyze_glm53_diagnostics.py   # GLM-5.3 length-stop and failure diagnostics
python refresh_v14.py                 # post-v13 three-seed aggregates and per-seed intervals → VALIDATION_V14_RESULTS.json
python refresh_v14_figures.py         # Figures 8–9 with the three-seed Phi-4/Aya replications
python analyze_paired.py              # run last: Table 5, then refresh_v14_tables.py → Tables 13–15
```

Running this sequence in this order regenerates `tables/main.tex`, `replicated.tex`, `paired.tex`, `scale.tex` and
`kolibri_languages.tex` byte-identical to the versions in `tables/` (checked 3 Nov 2026). `refresh_v12.py` and
`analyze_round20.py` write the v13 (single-seed) forms of Tables 13–14; the final `analyze_paired.py` call
updates them to the three-seed forms. Recomputed `VALIDATION_V14_RESULTS.json` can differ from the archived
copy in the last floating-point digit (summation order); every table is unaffected. The other tables are frozen; see `../docs/RESULTS_MAP.md`.
Do not run `legacy_generators/` or `refresh_v11.py`. They are kept for provenance, write older table
formats and are superseded by `refresh_v12.py`.

| Folder | Contents |
|---|---|
| `evidence/` | Archived reports, selected complete responses, harness outputs, configurations and campaign-log excerpts per round (`round4`…`round20`), plus `paired/`, `transfer_intervals/` and the significance CSVs. `post_v13/` holds the three-seed Qwen3-32B/GLM-4.7-Flash, Phi-4 mixed-format, Aya and Kolibri-1 evidence (reports, complete main-test responses, energy outputs, harness summaries, configs, scripts). `round20/manifest.json` and `post_v13/manifest.json` list SHA-256 hashes. |
| `data/` | Evaluation reports used by `refresh_figures.py` (rounds 4–19, eval60, packs). |
| `source_tables/` | CSV snapshot behind the frozen tables. |
| `tables/`, `figures/` | The LaTeX tables and figure files used in the accompanying study. |
| `VALIDATION_*.json`, `provenance.json` | Validation records and provenance of the table and figure build. |
| `make_figures_v08.py`, `sig.py`, `audit_round15.py` | Earlier generators and the round-15 audit, kept for provenance. |

### Manuscript clarity revisions (v14_2, v14_4; no new experiments)
- `tables/` now matches manuscript v14_4: captions clarified, labels renamed (`t:`), and the new `evaluation_map.tex` (origin, size, languages, purpose/RQ and results location of every evaluation set). `analyze_paired.py` and `refresh_v14_tables.py` write the same labels.
- Split audit (v14_2): `python3 audit_splits.py` checks the archived train (200 roots), validation (40 roots) and main-test (12 roots) splits in `../data/contrast13m_200` for overlap in example IDs, root IDs, semantic IDs and exact prompts (zero in all three pairs). `evidence/split_audit/SPLIT_AUDIT_V14_2.json` records rows, roots, families and SHA-256 of the audited files; `evidence/split_audit/curriculum.py` is the curriculum code that refuses test/diagnostic inputs. This is a split-leakage check, not a pretraining-contamination audit.
- `evidence/pack_documentation/` holds the manifests of the generated evaluation-only packs (core60/diag60 and the development112 coverage extension).
- `VALIDATION_V14_2.json`, `VALIDATION_V14_4.json`: build checks of the two revisions (all experimental values unchanged).

