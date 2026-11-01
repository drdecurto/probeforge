PROBEFORGE COVERAGE EXTENSION — CANDIDATE, NOT EXECUTED ON MODELS
Prepared 2026-10-13. Nothing was written to mn5_probeforge.

WHAT THIS ADDS
168 new root problems across seven additional task families, 24 roots per family.
English and Spanish versions, with genuinely reworded questions and structured
fact-order variants. There are 2,736 unique evaluation prompts.

Task families and why they matter:
* Reverse percentage: distinguish inverse operations from applying a discount.
* Conditional probability: use the conditioning population and ignore irrelevant
  counts; distinguish missing relevant and irrelevant numerical information.
* Modular schedule: simultaneous congruences, earliest feasible solution, and
  incompatible requirements.
* Inventory balance: operation order, conservation, and inability to fulfill
  a shipment when backorders/negative stock are explicitly forbidden.
* Dependency schedule: parallel tasks, critical path, and cyclic dependencies.
* Unit conversion: kg/g/mg, fractional quantities, conversion direction and scale.
* Linear optimization: finite maximum, empty feasible interval, and unboundedness.

These are additional task families relative to the existing evaluation, not
evidence that the base model has never seen the concepts. Some share mathematical
skills with the existing training families. This does not certify absence from
pretraining or provide a broad real-world reasoning benchmark.

PACK CONTENTS
development112.jsonl: 112 roots, 16/family, 1,824 prompts. Use for development.
confirmation56.jsonl: 56 disjoint roots, 8/family, 912 prompts. Freeze model and
decoding choices before evaluating. Do not train on or tune against this set.
pilot14.jsonl: 14 roots, 2/family, 228 prompts. A SUBSET of development112; never
count it as additional data or concatenate it with its parent set.
root_catalog.jsonl: root facts and cohort assignments.
answer_key.jsonl: exact expected outputs keyed by ID.
*_prompts_only.jsonl: ONLY ID and message list; safer exports for generic runners.
preview.txt: seven English examples and Spanish reformulations, with answers.
manifest.json / validation.json: counts, limitations, hashes and executed checks.
build_and_validate.py: deterministic standard-library-only builder. It writes
beside itself; run only in a separate output copy. It never accesses MN5 folders.

VIEWS AND PAIRED CONTROLS
Every root has base, paraphrase, fact_reorder, distractor, counterfactual,
missing_required, missing_optional and infeasible views, in both languages.
Linear optimization also has an unbounded view.

Unlike the historical prefix-style paraphrase, the new paraphrase actually
restructures the task statement. fact_reorder presents labeled facts in reverse
order. distractor adds irrelevant numbers 17 and 203.

Missing-required variants remove information needed to determine a result.
Missing-optional variants retain enough information: in probability, a large-ball
count is missing; elsewhere an explicitly irrelevant archive reference is missing.
Thus archive-reference variants are easy negative controls, not a comprehensive
test of subtle irrelevance. The ? marker remains: these cases do not establish
robustness to natural-language omissions without an explicit marker.

Counterfactuals change one fact and the exact solution. They are not matched
for arithmetic difficulty. Numeric ranges vary in most families; draw_scale is
a generation label, not an empirically calibrated difficulty rating.

Infeasible cases include impossible discounts, negative object counts/masses,
incompatible congruences, stock shortages, dependency cycles and empty intervals.
Negative counts/masses are deliberately contradictory inputs, not an estimate
of the prevalence or form of real-world errors.

STATUS COUNTS ACROSS BOTH FULL SETS
solved 2,016; needs_clarification 336; infeasible 336; unbounded 48.
Unboundedness is concentrated in one task family, so success does not establish
cross-family unboundedness recognition. Do not emphasize raw overall accuracy:
report per-family and per-status results, especially this minority status.

EXISTING COLLECTOR / VERIFIER COMPATIBILITY
Rows use kind=retention, family=retention_<task>, split=test. This uses the existing
exact-JSON branch without changing the original math solver. The word retention
is a TRANSPORT SCHEMA name here. This is a synthetic transfer/retention diagnostic,
not an independently sourced natural-language retention benchmark.

The required answer is only {"status":...,"answer":...}. Numeric answers must be
canonical reduced rational strings, e.g. "3" or "3/2". No explanation or facts keys
are permitted in this pack. Object key order does not matter; extra keys and
decimal alternatives do not pass the stock exact-match retention verifier.
This differs from the original four-key math protocol. Always compare base and
adapted models UNDER THIS SAME NEW PROTOCOL; do not compare absolute scores with
the original 41% result as if they measured the same task. Track format failures
separately if you analyze failures.

The full compatibility rows contain expected_response oracle metadata because
the stock retention verifier requires it. Send ONLY row.prompt to the model,
never the whole row, task_facts, or expected_response. Prompt-only exports exclude
oracle fields. They are not directly interchangeable with full collector inputs.

VALIDATION
Each generated semantic variant is computed by two arithmetic implementations:
inverse equations; explicit conditional sample spaces; CRT vs exhaustive modular
search; stock simulation; dependency graph traversal; independent conversion
tables; and interval endpoint optimization. This is a consistency check, not an
independent linguistic audit or formal proof.
All 2,736 gold outputs and wrong-status mutations were checked with the existing
ProbeForge v0.2 retention verifier; schema audit also passed.
Base and answer-preserving variants agree; changed-fact variants change the
answer; missing optional data keeps it; missing required data prevents a unique
determination; infeasible/unbounded views have the intended statuses.
Root cohorts are disjoint; meaningless archive references are removed during
semantic duplicate checks so ID changes cannot manufacture novelty.

STATISTICAL USE
Use root groups, not translated/perturbed prompts, as the resampling unit. Shared
templates still limit independence. Report each family separately and macro-average
family results; optimization has nine views while other families have eight.
Measure correct-to-wrong flips under paraphrase/reordering, joint paired
correctness, counterfactual sensitivity and clarification specificity.
When scoring the pilot, use pilot14 inputs; do not score it against development112,
since the stock evaluator counts missing requests as failures.

LIMITATIONS / STILL SCARCE
Only two languages, with authored translations not independently bilingual-reviewed.
Only two task formulations per language plus fact lists; no broad naturalistic
paraphrase coverage, native-speaker corpus or long context.
No external general-knowledge/open-ended retention suite, images, charts, tools,
multiturn interaction, adversarial safety evaluation or human preference labels.
No tokenizer preflight or model/GPU execution. Exact oracles do not verify
explanations because this protocol does not request them.

RUN PRIORITY
Start with pilot14 to check protocol compliance. Use development112 for diagnosis.
Once choices are frozen, evaluate confirmation56 and keep the reports separate.
Do not mix this pack into SFT/DPO: it is reserved for evaluating transfer.
Using the older non-thinking 512-token measurements, a crude linear extrapolation
for all 2,736 rows is ~3.7 GPU-hours at 8B or ~4.2 at 14B per checkpoint. Output
format and tasks differ, so pilot timing should replace those estimates.
