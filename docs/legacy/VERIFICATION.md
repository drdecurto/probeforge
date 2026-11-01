# Verification for ProbeForge 0.2

Executed in the development environment on 2026-09-06. These are software checks,
not evidence that a full-size model has improved.

- **30 automated tests passed.** Exact oracles, counterfactuals, units, strict JSON,
  leakage rejection, collection resume, mocked API retries and usage, mining,
  missing/truncated evaluation, retention import, LIC controls, vision annotation
  reflection, token masking/gradients and mocked Slurm submission were covered.
- **36,665 generated examples** across 1,400 root worlds and 12 disjoint families
  passed oracle-target and real Qwen3 tokenizer preflight. Longest sequence: 406
  tokens; mean: 291.52. Training: 26,665 rows; validation/test: 5,000 each.
- **Tiny random Qwen3 CPU run passed:** actual SFT, checkpoint resume, nonzero LoRA
  updates, frozen-reference caching, DPO, adapter reload/generation and BF16 merge.
- **Tiny random Qwen2.5-VL CPU run passed:** actual SFT with frozen visual encoder,
  image/processor handling, text-attention LoRA and adapter reload/generation.
- All four token-budget preparation arms and a 12-configuration experiment matrix
  were exercised with explicitly synthetic rollout fixtures. Small fixture budgets
  have visible token differences; inspect manifests for the actual experiment.
- Python/JSON files parsed; Slurm scripts passed shell syntax checks. The new Colab
  notebook was generated with no outputs, credentials or trained weights.

Detailed machine-readable reports are in verification/. Historical version-0.1
reports are under verification/legacy and are not presented as tests of new code.

Not run here: full-size 8B/3B training, CUDA/QLoRA, actual MareNostrum job submission,
live Nebius calls, GPU NVML hardware measurements, a Colab A100 session, or model
capability/energy improvements. Authored translations need bilingual review. VLM
experiments need your complete image annotations and split manifest. External
retention datasets and their ground truth must be supplied separately.
