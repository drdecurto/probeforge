# Method and data contract

## Concrete capability

The trained model receives a finite list of executable candidate rules, a common
initial observation when available, a menu of possible probes with integer costs,
three target inputs, and a remaining budget. It returns one JSON action:

```json
{"action":"probe","probe_id":"p3"}
```

or

```json
{"action":"answer","predictions":[3,9,15]}
```

The evaluator privately selects one candidate. A probe runs that actual candidate
and returns an observation. Success requires all three final outputs to match.
Parameters and candidate formulas are visible; the hidden index and target answers
are never sent to a policy. The runtime executes no model-written Python or shell.

This is an **active finite-hypothesis prediction** curriculum. An exact algorithm
can solve it. That algorithm is a useful teacher and upper/control baseline, not
a competitor we expect the LLM to beat on this closed task. The scientific question
is whether distilling the behavior into a language model yields useful improvements
and transfer. Real code repair or microgrid performance is not implied.

## Paired worlds and splitting

Eight worlds share the same public candidate ordering, observations, probe menu,
target inputs, and budget. Only the evaluator's hidden candidate differs. At least
some worlds require different target outputs. Duplicate target signatures can occur;
success concerns predictions, not hidden-index recovery.

The structural group ID depends on the family and unordered candidate parameter
triples. All worlds and variations of a parameter group stay in a single split.
Candidate order is shuffled independently of the true index. Model prompts contain
neither task IDs nor group IDs. Datasets include all eight worlds, so a constant
guess cannot exploit an imbalanced hidden-index distribution.

Train/validation/test-ID share six mechanism families but have disjoint parameter
groups. Test-OOD uses six different families. Small smoke datasets need not contain
every family in each split; use >=300 groups for the default experiment. Full
public-state equality within groups and file checksums are checked by `audit`.

## Exact teacher

Given current public observations, retain consistent candidates with a uniform
posterior. Define the stopping value as the fraction supporting the most frequent
target-output vector. Dynamic programming computes:

`V(state) = max(stop_success_probability, max_probe E[V(next_state)] - lambda*probe_cost)`.

The recursion is bounded by the remaining budget. The default lambda is 0.02.
Repeated observations are excluded, ties favor stopping, and no sampling of private
truth is needed to label an action. Exact teachers sometimes stop with unresolved
ambiguity; they do not receive an oracle final answer. This is deliberate.

SFT stores each visited decision as a public prompt plus one completion. Identical
prompt/completion rows are deduplicated. Preference rows compare actions under the
same prompt; differences <=0.01 are excluded. Two clear lower-value alternatives
are retained at most. These are exact-planner labels, not empirical student-action
values. Keep that distinction in experiment names.

## Student branching

Start from states actually visited by a student. Candidate actions include its
action, a stopping prediction computed from the public posterior, and sampled
probes. Branching samples plausible hidden worlds uniformly, applies each action,
then asks the same student to continue. Alternatives share world and continuation
seed. Value is mean final success minus lambda times all probe costs in the branch.

The current implementation does not penalize completion tokens in the training
reward. It caps them per decision and records them in evaluation. Tool cost and
token cost should therefore be reported separately. A future weighted-token reward
must be an explicit ablation, not silently described as already implemented.

At temperature zero, repeated continuations are deterministic for local inference;
set repeats=1. For stochastic continuation, increase repeats and inspect uncertainty.
The API seed only namespaces the response cache; provider reproducibility is not
assumed. Branch preferences with margins <=0.05 are dropped. This is a heuristic
noise filter, not a statistical confidence bound or causal-identification theorem.

## Parameter updates

All original backbone weights are frozen. LoRA is attached to attention and MLP
projection matrices, rank 16 by default. Qwen3-8B has about 43.6 million trainable
adapter parameters at this setting, roughly 0.53% of its 8.2B parameters. The run
manifest reports the actual number. Embeddings and the language-model head remain
frozen. QLoRA additionally quantizes the frozen backbone to NF4 on CUDA.

Only completion tokens and EOS are trained. The public prompt is masked with -100.
Examples exceeding max_length fail visibly; no answer or evidence is silently cut.
The Qwen chat template is applied with enable_thinking=False. The initial objective
is short, executable action selection; no teacher private reasoning is required.

DPO uses the standard loss:

`-log sigmoid(beta * [(log pi(chosen)-log pi(rejected)) - (log ref(chosen)-log ref(rejected))])`.

Log probabilities are summed over completion tokens, including EOS, and exclude
prompt/padding. The fixed reference is the SFT adapter. Reference values are cached
before DPO, allowing one resident backbone. Reference hashes cover weights,
tokenizer artifacts, row content, precision, quantization choice, and context limit.
Dropout is disabled for DPO. The code uses a small Transformers Trainer subclass
instead of depending on changing TRL dataset APIs.

## Extending to an open task

For arbitrary program repair, add an environment module with a strict public-state
serializer, real bounded tools, and an independent final evaluator. Keep hidden tests
separate from public probes. Generate problem-family splits before collecting traces.
Do not implement unrestricted `exec` inside the existing evaluator. An isolated
execution service would be a separate engineering component requiring validation.

For physical simulation, replace integer battery rules with your actual simulator,
define observable state and allowed interventions, and validate reward units and
feasibility checks. The collection, decision-row, training, and reporting interfaces
can be reused, but this release does not contain a real electrical simulator.

## Attribution

The checkpoint and curated training/evaluation package can be new contributions
without claiming that LoRA, DPO, active learning, or distillation are new ideas.
Relevant primary sources:

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Learning to Ask Informative Questions](https://arxiv.org/abs/2406.17453)
- [SCoRe](https://arxiv.org/abs/2409.12917)
- [Information Gain-based Policy Optimization](https://arxiv.org/abs/2510.14967)
- [Counterfactual Search Routing](https://arxiv.org/abs/2607.05752)
- [CausalFlow](https://arxiv.org/abs/2605.25338)

The proposed contribution should be supported by measured checkpoint gains,
reproducible artifacts, and transfer evidence. A new checkpoint name alone does not
establish a strong research result.
