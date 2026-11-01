---
language: [en, es, ca, ar, lb]
base_model: Qwen/Qwen3-8B
library_name: peft
tags: [lora, multilingual, reasoning, research]
---

# Fill after training: your verified-contrast model

Status: template only. No trained model or measured gain is represented here.

## Provenance

- Base model repository, immutable revision and license: TO FILL.
- Adapter/merged-weight checksums, training code revision and configuration: TO FILL.
- Dataset manifests, original source attribution and generated-data license: TO FILL.
- Unique training groups, repeated exposures, total sequence tokens: TO FILL.
- SFT/DPO seeds, reference checkpoint, quantization and software versions: TO FILL.
- Hardware, allocated GPU-hours, training/collection/evaluation cost: TO FILL.
- Teacher endpoint/model revision, accepted rows and review process if used: TO FILL.

## Intended capability and scope

The model is trained on verified structured answers across language and problem
variations. Interpretations of numerical facts and final answers are checked.
Free-form explanations are not automatically proof-verified. Record bilingual
template review and shared schema language explicitly. LIC-inspired synthetic
utility is not a calibrated human-behavior model. VLM adapters require a separate
vision base model and image-specific annotations.

## Evaluation to report

- Original checkpoint, translated SFT, random contrasts and mined contrasts.
- Matched output contract, decoding settings, sequence-token budgets and actual runtime.
- Per-language and worst-language accuracy; all-views-correct; boundary status results.
- Held-out family results, confidence interval unit and optimization seeds.
- Tokens, latency, generation energy coverage, and joules per correct answer.
- External retention benchmark names/revisions and regressions.
- For policies: validity, regret coverage, adaptive greedy and exact-search baselines.
- For VLMs: scene split, image hashes, annotation completeness, count/side-count results.

Do not present synthetic fixture scores, tiny random model execution, API latency,
or heuristic similarity as model capability, GPU energy, or correctness evidence.
Keep the base model's required attribution and applicable distribution terms.
