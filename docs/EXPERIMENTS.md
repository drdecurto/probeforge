# Experiment sequence

## Gate 1: software and data

Run unit/integration tests and the offline tiny-Qwen3 training smoke test. Audit
the dataset. Confirm all public paired-world prompts are identical, no groups
overlap splits, budgets are enforced, and exact-planner labels use only observations.
Look at a few stored decisions before training the 8B model.

## Gate 2: hardware pilot

Use 300 groups, 30 SFT updates, rank 16, context 2,048, batch 1, and the supplied
accumulation setting. Confirm finite loss, nonzero gradients, checkpoint save/load,
and acceptable memory/throughput. This is a smoke run, not a result to publish.

If the base already saturates the validation tasks, increase task difficulty or
move to a smaller baseline model before investing in training. If the model emits
invalid JSON frequently, inspect the chat template and token budget first.

## Gate 3: model comparison

| Run | Data and updates | Purpose |
|---|---|---|
| Base | No updates, identical tools | Measures actual adaptation gain |
| Exact SFT | Exact planner decision trajectories | Cheap starting checkpoint |
| Teacher SFT | Verified Nebius trajectories | Ordinary trajectory-distillation baseline |
| Mixed SFT | Matched subsets of both sources | Tests additional data value |
| Exact SFT + exact DPO | Planner action preferences | Tests preference stage |
| Exact SFT + branch DPO | Student-continuation preferences | Tests current-student action supervision |

For a teacher-only baseline, train on `data/nebius_teacher/sft_successes.jsonl` with
the same validation file. Use `mix --cap-per-source` or explicitly sampled manifests
to control example counts. Count tokens too: equal example counts do not ensure
equal training compute. Avoid interpreting a larger dataset as a mechanism gain.

The `random` symbolic backend chooses random probes but predicts using exact
arithmetic afterward. It is a controlled task baseline, not an ablation of the
learned model. To test randomized probes in the learned model itself, use the
`random_probe` intervention described below; keep the same stopping/answer policy.

```bash
probeforge evaluate --tasks data/tasks/test_ood.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --intervention random_probe --out runs/sft_random_probes --budgets 0 1 2 4
```

This replaces only a valid selected probe with a uniformly random available probe
of the same cost. Stopping predictions, decoding, and tools remain identical. The
original proposed action is recorded alongside the executed replacement. The
random choice may coincide with the original action.

## Gate 4: final evaluation

Choose settings on validation only, then freeze them. Evaluate test-ID and test-OOD
at budgets 0, 1, 2, 4. Use exactly the same sampled groups, seeds, decoding, and
precision in compared runs. Report:

- Exact correctness of the three target predictions.
- Fraction of counterfactual groups for which every world is solved.
- Tool calls, consumed tool budget, prompt tokens, generated tokens, and latency.
- Invalid JSON/action rate, including unaffordable or repeated probes.
- Group-bootstrap confidence intervals and paired differences.
- Training, reference, generation, and evaluation GPU-hours; Nebius input/output tokens.

The all-world metric measures evidence sensitivity jointly with arithmetic and
action validity. It is not a pure causal-reasoning metric. Token counts for API
backends depend on the provider returning usage fields. The episode availability
flag records missing usage; summary token means are null unless usage is complete.
Symbolic policies have no LLM token usage and likewise report null token means.

Use at least three optimization seeds for promising settings. Report the seed
spread separately from bootstrap uncertainty across task groups. The bootstrap
does not account for unobserved training seeds or establish real-world transfer.

The current tasks do not solicit an initial tentative answer, so the repository
does not report correct-to-incorrect revision rates. Add a separately controlled
two-answer evaluation if that becomes a research claim.

## Interpretation

A useful result is improved held-out verified success at a matched tool/token
budget, or lower cost at comparable success. Predetermine what gain is worth further
GPU time. Do not invent a promised percentage improvement before running the base.

If improvements stay within familiar arithmetic families, describe the release as
a specialized model. Claims about arbitrary code repair, broad scientific reasoning,
or realistic microgrids require additional environments and external benchmarks.
The release does not include pretrained benchmark scores or Nebius-generated data.
