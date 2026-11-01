# Release verification — 6 September 2026

This report describes executed software checks. No trained 8B checkpoint or
measured capability gain is included in this release.

## Passed

- 15 unit/integration tests using Python 3.12, including actual local HTTP requests
  to a mock OpenAI-compatible server, cache reuse, and API request-budget enforcement.
- Dataset reproducibility, checksum audit, disjoint group splits, identical paired
  public prompts, and private-truth exclusion from prompts.
- Exact-planner behavior, runtime budget enforcement, malformed-action rejection,
  collection, counterfactual branching, evaluation, and paired comparisons.
- Completion masking, EOS handling, refusal of silent truncation, padding-invariant
  preference log probabilities, and gradient flow.
- Slurm submission argument test using a fake scheduler (no real job submission),
  plus Bash syntax checks for every shell/Slurm script.
- Real CPU training of a tiny, randomly initialized **Qwen3ForCausalLM**: SFT weight
  updates, periodic checkpoint save, resume from the latest checkpoint, fixed-reference
  probability caching, DPO updates, adapter reload, generation, and merged export.
- Real Qwen3-8B tokenizer test on 155 generated decisions: maximum 560 tokens,
  mean approximately 499, within the configured 2,048-token limit. This is a sample
  check; `preflight` validates every row of the user's actual train/validation files.
- Colab notebook JSON and all nine Python code cells parse successfully.
- The documented `scripts/prepare_data.sh` completed on the default 300 groups:
  1,840 unique SFT training decisions, 2,817 training preference pairs, and 298
  validation decisions. Symbolic baseline evaluation completed. Generated scores
  belong to the exact algorithm, not a language model.
- Full tokenizer preflight passed on all 2,138 train/validation decisions from that
  pilot: maximum 560 tokens, mean approximately 499.3.

Core training versions tested: Torch **2.6.0+cpu**, Transformers **4.51.3**, PEFT
**0.15.2**, Accelerate **1.6.0**. The initial CUDA wheel could not load in this CPU
execution environment; verification used the official CPU wheel instead.

Machine-readable tiny-model training and tokenizer results are in `verification/`.
These losses and throughput numbers belong to a tiny random CPU model and must
never be cited as Probe-8B results or estimates of H100 throughput. The DPO smoke
test checks updates and execution, not improvement on an eight-example dataset.

## Not executed

- Full Qwen3-8B training, NVIDIA CUDA/QLoRA memory or speed measurements.
- Actual MN5 Slurm jobs, BSC module/container integration, or multi-GPU NCCL training.
- Live Nebius calls, billing, provider-specific reasoning controls, or available
  model IDs for your account. The local HTTP mock tests only the request contract.
- Execution in Google Colab or on Apple M3/MPS.
- Broad reasoning benchmarks, realistic microgrid control, or arbitrary code repair.

A two-process CPU torchrun check was attempted but the environment denied Gloo's
network-device initialization (`Operation not permitted`). No distributed-training
success is claimed. The one-process training path passed, and the Slurm launch
structure is tested without contacting a scheduler. Run the documented 30-step
pilot on your allocated hardware before a larger experiment.
