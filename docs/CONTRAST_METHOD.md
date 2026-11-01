# Verified contrast training

This release implements the proposal from the evaluation-repository analysis.
It is an experimental post-training pipeline. It contains no trained 8B weights
and does not establish a capability improvement or novelty priority.

## Mechanism

1. Generate latent mathematical worlds with exact rational answers.
2. Render English, Spanish, Catalan, Arabic, and Luxembourgish questions.
3. Keep base, paraphrase, distractor, counterfactual and boundary views together.
   Rate and energy families also contain a unit-conversion view.
4. Collect the actual student checkpoint. Select completely observed groups with
   mixed success or failures on changed constraints. API failures are missing
   observations, not reasoning mistakes. Truncation remains a failed budgeted response.
5. Train on independently generated exact targets. This first release uses the
   student to select data; it does not need a large teacher to create mathematical gold.
6. Optionally train correctness DPO using actual complete student failures and
   fixed SFT-reference log probabilities. Verified brevity candidates are written
   to a separate review file; they are not silently included in correctness DPO.

The base model, translated-SFT baseline and selected curriculum must be evaluated
under the same output contract, context limit, decoding mode, budgets, and test
examples. A baseline prompted differently would confound the training comparison.

## What is verified

Math outputs contain `status`, `answer`, `facts`, and `explanation`.
Statuses are `solved`, `needs_clarification`, `infeasible`, and `unbounded`.
Answers and interpreted facts are compared with exact rational arithmetic. Signs
and units matter; booleans, duplicate JSON keys and nonfinite values are rejected.
This checks the final answer and interpreted formulation. **It does not verify a
free-form proof or explanation.** The authored explanations are intentionally short.
Unspecified variance convention is explicitly a request for clarification. An
underdetermined equation requests the information needed for a unique answer.

The natural-language templates are authored draft translations and require bilingual
review. Shared JSON keys and a small amount of English schema specification remain
in every language. Results therefore measure multilingual question interpretation
under a shared formal response schema, not entirely native-language reasoning.
Template explanations are no evidence of fluent explanation quality.

No arbitrary Python, notebook or model-generated program is executed as a verifier.
Production optimization enumerates a bounded integer feasible set. The cubic
optimization oracle explicitly distinguishes nonnegative and unrestricted domains.

## Leakage controls

Training families: rate, affine equation, weighted mean, variance, cubic product,
and battery energy. Validation families: joint work rate, rectangle optimization,
and two-equation systems. Test families: two-leg journeys, integer production,
and compound interest. These are structurally related competencies but disjoint
authored templates. They are not evidence of universal domain generalization.

All translations, perturbations and descendants of a root world stay in one split.
Preparation rejects test/diagnostic inputs and cross-split root or family overlap.
VLM splits use scene identities and image hashes, since the same task is evaluated
on unseen scenes. Near-duplicate scenes still need dataset-level review.

The published 79 questions are **diagnostics only**. `import-legacy` retains
references as unverified and never promotes heuristic scores to training labels.
The old benchmark's public availability also prevents claiming it is contamination-free.

## Resource-matched experiments

`prepare --token-budget N --tokenizer PATH` counts actual prompt-plus-completion
tokens and selects/repeats complete root groups until another group cannot fit.
It records actual exposure, unique instances and token count. Whole groups cause
small budget differences: inspect the matrix manifest. Run exactly one epoch,
`max_steps=-1`, with identical optimizer and batch settings across arms.
Sequence-token matching is not identical GPU compute: padding, attention length,
quantization and hardware matter. Report measured training runtime/GPU-hours too.

Translated SFT uses base views in all languages. Random uses all views without
mining. Minimized-counterfactual ablation (`no_counterfactual`) excludes boundary,
counterfactual and rewired views and uses random selection; it tests view inclusion
against random, not an isolated mining-by-view interaction. For the latter, add
the same eligibility restriction to mining in a separately documented experiment.

## Metrics

Pass@1 uses sample zero at each budget; additional samples support preference
construction and are not silently treated as additional independent problems.
Missing, error and length-stopped responses count as failures. Reports include
per-language, worst-language, per-view and per-family scores, and the fraction of
complete root groups where every view is correct. Output-language fidelity and
unrelated-task retention require additional external evaluation; they are not
inferred from these exact-answer scores.

Confidence intervals resample root worlds, keeping views together. They are
conditional on the small fixed set of families, not uncertainty over all unseen
families. Model comparisons require identical dataset hashes. Prefer three
optimization seeds after the pilot establishes an effect. Keep the data-generation
seed fixed for that comparison and vary it separately later.

## Energy

Local collection warms the model, synchronizes CUDA, and measures only generation
on an explicitly identified physical GPU. NVML total-energy counters are preferred;
sampled power uses trapezoidal integration as fallback. CUDA/NVML UUIDs must agree.
No energy is reported for remote APIs, missing meters or incomplete energy coverage.
Measurements include device idle power and are valid only when the measured GPU
is not shared with another workload. Short calls may be below counter resolution.

The report counts actual attempts, records unknown usage after errors, and shows
joules per correct answer alongside accuracy. Remote server-side compute and
energy cannot be reconstructed from local wall time. The older energy script's
three-runs-per-problem denominator must not be interpreted as one completion.

## Attribution and source audit

- Cross-platform study: https://doi.org/10.1016/j.ipm.2026.104878
- Multilingual LLM energy: https://doi.org/10.3390/electronics15071395
- Multilingual VLM energy: https://doi.org/10.3390/drones10050361
- LIC: https://doi.org/10.3389/frai.2026.1820827
- MAPO: https://arxiv.org/abs/2401.06838
- STaR: https://arxiv.org/abs/2203.14465
- R3S: https://arxiv.org/abs/2602.05940

At the pinned source revisions, the two English result repositories have 79
distinct question strings and 4,310 distinct question-response pairs after exact
deduplication. The original crossplatform response pairs are contained in the
comparison release. The energy repository contains 65 aggregate records and
12 translated 79-question lists plus English, but no released raw multilingual
completion logs were found. The VLM scorer uses generic English keyword/reference
targets rather than per-image gold. These observations motivated new exact labels.

The LIC paper attributes its small rewire advantage to refreshing target identities,
with essentially unchanged policy parameters. Our separate numerical coverage
curriculum explicitly changes benefits/fatigue and includes adaptive controls.
It is not a reproduction of the published LIC simulator or evidence about humans.
