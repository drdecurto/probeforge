PROBEFORGE — CANDIDATE EVALUATION EXTENSION
Prepared 2026-10-13. No MN5 job has been run with this data.
The experiment folder was read only and has not been modified.

PURPOSE
Increase coverage beyond the existing 12 root problems, while separating
confirmation on new numbers from targeted diagnostic exploration. This package
adds 120 distinct root problems: 40 journey, 40 production, 40 interest.
There are still only THREE problem families. It is not a general reasoning
benchmark or an unrelated retention benchmark.

WHAT TO USE
test_core60.jsonl: 60 roots, 20/family, 6,240 prompts. Fresh candidates from the
historical numerical generator after semantic duplicate exclusion. Reserve for
confirmation after choices are frozen. Core base and counterfactual construction
retain the historical generator. Boundary construction differs as described below.

test_diagnostic60.jsonl: 60 different roots, 20/family, 6,240 prompts. Use for
development and diagnosis, not as a final untouched test after inspecting results.

pilot_diagnostic12.jsonl: 12 roots, 4/family, 1,248 prompts, one root per diagnostic
stratum. This is a SUBSET of diagnostic60. Never concatenate it with diagnostic60
or report it as 12 additional problems. It is a cheap smoke/pilot set, not the
solution to the small-sample problem.

Each full set uses 13 languages and 8 views: base, paraphrase, distractor,
counterfactual, boundary, context_academic, context_business, expansion.
Total unique prompts across core60 and diagnostic60: 12,480.

DIAGNOSTIC DESIGN: FIVE ROOTS PER STRATUM
Journey: equal travel time with integer answer; unequal travel times;
unequal distances with repeating-fraction answer; equal distances/harmonic mean.
Production: unique optimum; multiple profit-maximizers with explicit tie rule;
binding minimum-production constraint; optimum that leaves unused capacity.
Interest: integer balance; terminating-fraction balance; repeating-fraction
balance; longer compounding horizons of 5–8 years. Rates vary beyond the old 5%.

Diagnostic counterfactuals change one fact, must change the exact solution, and
preserve the answer's integer/terminating/repeating number class. This is only
a PARTIAL difficulty control: digit counts, intermediate operations, denominators,
and human/model difficulty are not fully matched. Interest changes can be large
to preserve number class. Do not claim these are perfectly difficulty-matched pairs.

BOUNDARY DESIGN AND COMPARABILITY
The old journey/interest boundaries removed distance_a/principal. This can reduce
many distinct roots to identical boundary questions already in historical data.
Core journey boundaries instead remove speed_a; core interest boundaries remove
years. Core production retains the historical impossible minimum of 1000.
Diagnostic journey alternates missing speed and zero speed. Diagnostic production
sets minimum=floor(capacity/2)+1, just beyond the feasible item-count limit.
Diagnostic interest alternates missing rate and missing years.
All 120 boundary semantic worlds are distinct and absent from the exclusion list.
Because boundary design differs, report boundary accuracy separately when
comparing to historical reports; do not attribute all aggregate differences to
more test problems. No unbounded cases are manufactured for these three families.

EXACT ANSWERS AND COMPATIBILITY
JSONL rows follow the existing math evaluation schema, use split=test, and retain
the existing JSON keys and canonical-units suffix. Metadata fields identify suite
and stratum. Gold answers are in answer_key.jsonl, separate from model prompts.
The world field is oracle metadata already present in ProbeForge's schema; send
only the prompt messages to the model, never whole rows or answer keys.
This is evaluation-only data. Do not feed it to prepare, SFT, DPO, or mining.

Reference copies of the local v0.2 dataset/verifier/io and v0.2.1 worlds/languages
are included with the original LICENSE and NOTICE for reproducibility.
The existing 13-language authored templates are reused; this is not a new
bilingual translation review. The paraphrases remain wrappers around existing
templates. Check translations before strong language-specific claims.

CHECKS COMPLETED
1. 120 unique roots; 12,480 unique example IDs; balanced family/language/view counts.
2. Base, changed and boundary worlds excluded against locally archived datasets
   and a reconstruction of the documented seed-42, groups=1000 generator.
   See exclusion_scope.json for the exact scope. No live cluster was inspected.
3. A second mathematical implementation agrees with the existing exact oracle:
   cross-multiplied travel formula, a y-first production optimizer, and repeated
   annual interest updates. This checks arithmetic, not natural-language fidelity.
4. All 12,480 gold responses pass the existing verifier. Deliberately perturbed
   answers/statuses fail for all 12,480 examples.
5. Equivalent views retain the same world/answer; every counterfactual changes
   the exact solution; diagnostic number-class constraints are checked.
6. Schema audit passes. File SHA-256 hashes are recorded in manifest.json.

NOT YET CHECKED
No model responses, tokenizer preflight, GPU runtime, live-cluster overlap audit,
or independent bilingual review. Exact novelty against this inventory does not
establish absence from pretraining or other uninspected datasets. Shared family
and wording templates remain and limit statistical generalization.

RECOMMENDED USE
Start with pilot_diagnostic12 for infrastructure and failure analysis. When ready,
evaluate all diagnostic60 for development. Freeze model/checkpoint/decoding choices
before core60 evaluation. Keep core and diagnostic reports separate.
Use the exact input file corresponding to collected IDs when scoring; the old
evaluator treats uncollected rows as failures. Do not score a pilot against a
full 60-root file. The pilot is contained within the diagnostic file.

Report full correctness, exact-answer correctness, fact extraction, status,
solved-only accuracy, per-family/stratum metrics, and conditional consistency.
Bootstrap at root level, not prompt level; report seed variation separately.
Also examine shared numerical/template structure when interpreting uncertainty.

Rough historic evaluation cost per checkpoint at 512 non-thinking tokens:
pilot12: about 1.7 GPU-hours for 8B or 1.9 for 14B;
one 60-root set: about 8.5 GPU-hours for 8B or 9.5 for 14B.
Both 60-root sets: about 17/19 GPU-hours respectively. These are extrapolations,
not measured costs for the new problems; allow extra margin for longer outputs.

FILES FOR REVIEW
preview.txt: all 12 pilot roots in English, with base/counterfactual/boundary answers.
root_catalog.jsonl: all 120 roots with design metadata.
answer_key.jsonl: gold response per example ID, including all languages/views.
manifest.json and validation.json: counts, design limitations, hashes and checks.
build_and_validate.py: deterministic standard-library-only regeneration script.
It writes beside itself; run only in a separate output copy. It does not access
the experiment folder. The packaged exclusion inventory is its only prior-data input.
