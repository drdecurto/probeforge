# ProbeForge / ContrastForge

Verified multilingual contrast post-training


ProbeForge/ContrastForge generates mathematical *problem worlds* with exact rational solvers, renders each
world in 13 language configurations and eight *views* (answer-preserving rewrites, a missing-fact/infeasible
boundary view and a counterfactual), verifies model responses exactly, and turns the same contrast groups into
supervised training curricula (random, error-mined, status-diverse, mixed-format). The study adapts
Qwen3-8B/14B/32B, Phi-4, Phi-4-mini, Aya-Expanse-8B, GLM-4.7-Flash and Kolibri-1 with LoRA/QLoRA under a nominal
three-million-token budget, and evaluates transfer to a published multilingual reasoning benchmark and to
Belebele, Global-MMLU-Lite and MGSM.

Every reported number in the accompanying study can be traced to a file in `paper/evidence/`, `results/` or `data/` and to a
script in `paper/`; see [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/contrastforge/` | Problem worlds, 13-language renderer, views, exact verifier, curriculum builder (random/mined/translated/no-counterfactual), collection (local vLLM/HF or OpenAI-compatible API), evaluation and paired comparison. |
| `src/probeforge/` | Training (LoRA/QLoRA SFT, cached-reference DPO), tokenisation with completion-only labels, I/O and hashing utilities. |
| `configs/` | Every training configuration used in the accompanying study (`v2_*`, `v14b_*`, `v3_*`, `vphi4*`, `v4*`, `v4c_*`, `vaya_*`, `v32b_*`, `vglm_*`, `vkol_*`). |
| `slurm/`, `scripts/`, `tests/`, `examples/`, `notebooks/` | Generic launchers, helpers, unit tests (29 pass), small previews and Colab notebooks. |
| `campaign/` | The scripts actually run on MareNostrum 5: mixed-format data (GSM8K translation/verification, mixtures), GRPO, the energy-benchmark evaluators, lm-evaluation-harness wrappers and patch, Qwen3-32B / GLM-4.7-Flash / Kolibri-1 / GLM-5.3 / Muse-Spark serving, and the Slurm chains. See [`campaign/README.md`](campaign/README.md). |
| `data/` | Generated train/validation/test sets, held-out evaluation packs (core60, diag60, development112), energy-benchmark problems, the verified multilingual GSM8K pool, curriculum manifests and base-model mining rollouts. See [`data/README.md`](data/README.md). |
| `results/` | Evaluation reports for all arms, energy-benchmark outputs, lm-evaluation-harness results and significance tables. See [`results/README.md`](results/README.md). |
| `paper/` | Reproduction package for the accompanying study: archived evidence, analysis/table/figure generators, generated tables and figures. See [`paper/README.md`](paper/README.md). |
| `docs/` | Method, runbooks, environments, reproduction guide and the results map. |

No model weights or adapters are included (see *Models* below).

## Quick start (CPU, no GPU needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q tests                       # 29 passed, 1 skipped
contrastforge demo --out demo_run     # synthetic end-to-end demo (fixture responses, not LLM results)
```

Score any set of responses against the main test with the exact verifier:

```bash
contrastforge evaluate --inputs data/contrast13m_200/test_eval12.jsonl \
  --rollouts paper/evidence/round20/q32/eval/test_v4c32b.jsonl --out /tmp/report.json   # pass@1 0.450 (Qwen3-32B mixed-format, seed 1)
```

## Reproducing the results

* **Tables and figures from archived results (minutes, CPU):** `cd paper && pip install numpy matplotlib reportlab`
  then follow [`paper/README.md`](paper/README.md). `analyze_paired.py`, `analyze_round20.py`,
  `analyze_transfer_intervals.py`, `analyze_glm53_diagnostics.py` and `refresh_v14.py` recompute Tables 5, 13–15, the transfer
  intervals, the per-seed paired intervals and the GLM-5.3 diagnostics from raw responses and check them against the archived reports.
* **Re-running experiments (GPU cluster):** [`docs/REPRODUCING.md`](docs/REPRODUCING.md) gives the full
  sequence — generate data, collect base-model rollouts, build curricula, train, collect and evaluate, external
  benchmarks — with the exact configurations and environment versions in [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md).

## Main results

Exact structured-response accuracy on the 1248-prompt held-out test (12 roots × 13 languages × 8 views,
non-thinking, 512 tokens):

| Model | Base | Best adapter (curriculum) | Seeds |
|---|---|---|---|
| Qwen3-8B | 0.036 | 0.318 ± 0.033 (mined) | 3 |
| Qwen3-14B | 0.083 | 0.413 ± 0.005 (random) | 3 |
| Phi-4 | 0.022 | 0.341 ± 0.032 (random) | 3 |
| Qwen3-32B (QLoRA) | 0.144 | 0.407 ± 0.040 (mixed-format) | 3 |
| GLM-4.7-Flash (MoE) | 0.056 | 0.311 ± 0.010 (mixed-format) | 3 |
| Kolibri-1 (MoE) | 0.079 | 0.416 ± 0.030 (random); 0.415 ± 0.053 (mixed-format) | 3 |

Unadapted references under a 4096-token low-effort protocol: Muse-Spark-Open 0.974, GLM-5.3 0.417.
See the accompanying study for transfer, paired-consistency, thinking-mode, energy and standard-benchmark results and
their limitations.

## Models

Base checkpoints are downloaded from their official Hugging Face repositories and remain under their own
licenses. Trained adapters are not part of this repository; they can be regenerated with the configurations in
`configs/` (training takes 1–2 GPU-hours per arm at 8–14B on H100; Kolibri-1 needs four H100 per arm for 1.5–2 h).


## License

Code and authored data: MIT ([`LICENSE`](LICENSE)). Third-party data and model terms: [`NOTICE`](NOTICE).
