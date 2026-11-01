# 0.2.0

Adds `contrastforge` alongside the retained `probeforge` pipeline:

- Exact multilingual worlds and same-meaning/changed-meaning contrast groups.
- Assumption, infeasibility and unboundedness targets; strict facts/answer grading.
- Resumable local/API generation with explicit completion and energy accounting.
- Failure-based curricula, baseline arms, whole-group token budgets and actual-error DPO.
- Paired held-out metrics, external exact-JSON retention tasks and source diagnostics.
- LIC-inspired bounded policy optimization with adaptive controls.
- Optional Qwen2.5-VL annotated-image training with frozen vision and reflection labels.
- Single-process Slurm launcher/matrices, new Colab notebook and execution runbook.
- Stronger training-resume checks for changed datasets and protected configuration.

This release does not train or include full-size model weights. It preserves the
original implementation and its historical reports separately for reproducibility.
