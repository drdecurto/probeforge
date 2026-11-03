# Campaign scripts (as run on MareNostrum 5)

These are the exact scripts used for the experiments. They hard-code the cluster's scratch root
(`S=/gpfs/scratch/...`), Slurm account and QoS; adapt them before reuse. Generic, portable entry points are the
`contrastforge` CLI and `slurm/` (see `../docs/REPRODUCING.md`).

| Folder | Purpose |
|---|---|
| `v4/` | Mixed-format data: `gsm_ml.py` (translate GSM8K training problems, keep number-preserving translations, generate and verify solutions), `v4_mix.py` (contrast JSON / prose / GSM8K mixture by target-model tokens), `it_diag.py` (Italian/Portuguese yield diagnostic that exposed the literal answer-placeholder issue fixed in v4c). |
| `grpo/` | GRPO with the exact verifier as reward (TRL 0.18.2): `prefilter.py` (mixed-difficulty prompt selection with vLLM), `grpo_train.py`. Reported as a null result in the campaign log; not part of the accompanying study. |
| `energybench/` | Evaluators for the published benchmark: `eval_patched.py` (raw completion), `eval_chat.py` (chat template, thinking disabled), `answer_acc.py` (key-value match), `node_run2.sh` (multi-slot runner). |
| `stdbench/` | lm-evaluation-harness runner `node_std.sh`, model list, task list, `aggregate.py`, adapter merge helper and the vLLM-0.32 compatibility patch `lm_eval_vllm_causallms.patch`. |
| `q32/` | Qwen3-32B: vLLM TP4 server with both adapters, sharded API collection (`remaining.py`, `api_extra.json`), energy clients, adapter merge and lm-eval. |
| `glm/` | GLM-4.7-Flash: environment, adapter merge, per-model serving/evaluation and lm-eval. |
| `kolibri/` | Kolibri-1: LoRA merge into the original BF16 shards (`merge_lora.py`, keeps the plugin's key names), per-node vLLM 0.29 + `aleph-alpha-inference` server with main-test API collection, energy clients and lm-eval `local-completions` standard benchmarks (`serve_eval.sh`), environment (`env.sh`). |
| `glm53/`, `muse/` | 4-node serving (TP4 × PP4) and evaluation chains for the unadapted references. |
| `sig/` | Significance analysis (`sig.py`); per-seed paired energy bootstraps for the Phi-4/Aya replications (`r4_energy.py`) and Kolibri-1 (`kol_energy_boot.py`). |
| `chains/` | All Slurm batch files and detached chain scripts (collection, training, evaluation, energy and standard benchmarks, downloads), in the order they were used. Post-v13 additions: `master_1101.sh` (Qwen3-32B and GLM-4.7-Flash seeds 2–3), `master_r4.sh` (Phi-4 mixed-format and Aya seeds), `kolprobe_chain.sh`, `kolsft_chain.sh`, `kolpost_chain.sh` (Kolibri-1 probe, training, merge and evaluation) and their batch files. |
