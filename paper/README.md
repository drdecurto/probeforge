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
```

Running this sequence regenerates `tables/main.tex`, `replicated.tex`, `paired.tex` and `scale.tex`
byte-identical to the versions in `tables/` (checked 1 Nov 2026). The other tables are frozen; see `../docs/RESULTS_MAP.md`.
Do not run `legacy_generators/` or `refresh_v11.py`. They are kept for provenance, write older table
formats and are superseded by `refresh_v12.py`.

| Folder | Contents |
|---|---|
| `evidence/` | Archived reports, selected complete responses, harness outputs, configurations and campaign-log excerpts per round (`round4`…`round20`), plus `paired/`, `transfer_intervals/` and the significance CSVs. `round20/manifest.json` lists SHA-256 hashes. |
| `data/` | Evaluation reports used by `refresh_figures.py` (rounds 4–19, eval60, packs). |
| `source_tables/` | CSV snapshot behind the frozen tables. |
| `tables/`, `figures/` | The LaTeX tables and figure files used in the accompanying study. |
| `VALIDATION_*.json`, `provenance.json` | Validation records and provenance of the table and figure build. |
| `make_figures_v08.py`, `sig.py`, `audit_round15.py` | Earlier generators and the round-15 audit, kept for provenance. |
