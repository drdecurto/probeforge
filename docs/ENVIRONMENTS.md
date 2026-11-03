# Software environments

All runs used MareNostrum 5 (BSC), accelerated partition: nodes with 4 × NVIDIA H100 (64 GB), Slurm,
`acc_debug` QoS (2-hour jobs). Compute nodes have no internet access; models, datasets and Python packages
were staged from the login node. Seven Python environments were used (versions recorded with
`importlib.metadata` on 31 Oct 2026; `kolvllm`/`koltrain` on 2 Nov 2026):

| Env | Used for | torch | transformers | peft | other |
|---|---|---|---|---|---|
| `cf` | All LoRA/QLoRA SFT and DPO for Qwen3-8B/14B/32B, Phi-4, Phi-4-mini, Aya; local-backend collection | 2.6.0 | 4.51.3 | 0.15.2 | accelerate 1.6.0, bitsandbytes 0.45.5 |
| `vllm32` | vLLM serving (Qwen3-32B, GLM-4.7-Flash, GLM-5.3), energy-benchmark generation | 2.13.0 | 5.19.0 | – | vllm 0.32.0 |
| `lmeval` | Belebele / Global-MMLU-Lite / MGSM | 2.13.0 | 5.19.0 | 0.21.2 | vllm 0.32.0, lm_eval 0.4.9.1 (+ patch below), datasets 3.6.0 |
| `grpo` | GRPO experiment (layered on `cf`) | 2.6.0 | 4.51.3 | 0.15.2 | trl 0.18.2, datasets 3.6.0 |
| `glm` | GLM-4.7-Flash LoRA (layered on `vllm32`) | 2.13.0 | 5.19.0 | 0.18.1 | accelerate 1.15.0 |
| `kolvllm` | Kolibri-1 serving | 2.13.0 (cu130) | 5.22.1 | – | vllm 0.29.0, aleph-alpha-inference 1.0.0 (vLLM plugin) |
| `koltrain` | Kolibri-1 LoRA (layered on `vllm32`) | 2.13.0 (cu130) | 5.20.2 | 0.18.1 | – |

`pip install -e '.[train,quant]'` reproduces `cf`; `pip install -e '.[train5]'` on top of `vllm==0.32.0`
reproduces `glm`.

## Notes that matter for reproduction

* **lm-evaluation-harness 0.4.9.1 with vLLM 0.32** needs the compatibility patch
  `campaign/stdbench/lm_eval_vllm_causallms.patch` (optional `ray`, tokenizer/port helpers moved in vLLM,
  engine kwargs filtered against `EngineArgs`). Set `NUMEXPR_MAX_THREADS=64`.
* **Thinking mode.** Qwen3 and GLM-4.7-Flash are evaluated with `enable_thinking=False`
  (`chat_template_kwargs` in API calls; `campaign/q32/api_extra.json`). Omitting it makes Qwen3-32B think
  until the 512-token cap (all answers fail). GLM-5.3 and Muse-Spark always reason; they use
  `reasoning_effort` low/max and a 4096-token budget.
* **transformers 5** removed `TrainingArguments(warmup_ratio=…)`; `src/probeforge/train.py` maps it to a
  float `warmup_steps` and drops unsupported arguments, so the same code runs under 4.51 and 5.x.
* **Resuming training** restores `save_steps`/`eval_steps` from `trainer_state.json`, not from the config.
  Long runs therefore use `eval_steps` ≫ steps and `final_eval: false` (configs `v32b_*`, `vglm_*`).
* **Multi-GPU models:** `device_map`/`max_memory_gib` in a config enables naive model parallelism
  (GLM-4.7-Flash, 2 GPUs per arm; Kolibri-1, 4 GPUs per arm with `max_memory_gib` 56).
* **GLM-5.3 on 4 nodes (vLLM mp backend):** set `VLLM_HOST_IP` per node, `--enforce-eager`, and
  `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` on offline nodes (FlashInfer otherwise downloads FP8 kernels at
  run time). See `campaign/glm53/` and `paper/evidence/round20/SERVING_NOTES.txt`.
* **Kolibri-1.** The checkpoint needs the `aleph-alpha-inference` plugin for vLLM 0.29. LoRA targets
  attention and shared-expert projections only (routed experts are fused 3-D parameters and stay frozen).
  Adapters are merged into the original BF16 shards with `campaign/kolibri/merge_lora.py` before serving.
  The weights take about 11.5 min to load from GPFS, so server-readiness waits must exceed 15 min. The
  plugin is incompatible with lm-eval's in-process vLLM backend; standard benchmarks use lm-eval
  `local-completions` against the server (`campaign/kolibri/serve_eval.sh`), which needs the `[api]` extra
  (`tenacity`, `aiohttp`) in the `lmeval` environment.
* **Energy-benchmark clients:** run one CPU thread per scoring client
  (`OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_MAX_THREADS=1`) when many languages are scored in parallel.
