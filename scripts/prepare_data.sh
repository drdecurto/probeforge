#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PF_PYTHON="${PF_PYTHON:-python}"
"$PF_PYTHON" -m probeforge.cli generate --out data/tasks --groups "${PF_GROUPS:-300}" --seed 42
"$PF_PYTHON" -m probeforge.cli audit --data data/tasks
"$PF_PYTHON" -m probeforge.cli build --tasks data/tasks/train.jsonl --out data/expert_train
"$PF_PYTHON" -m probeforge.cli build --tasks data/tasks/validation.jsonl --out data/expert_validation
"$PF_PYTHON" -m probeforge.cli evaluate --tasks data/tasks/test_ood.jsonl --backend oracle --out runs/oracle_ood --budgets 0 1 2 4 --no-traces
"$PF_PYTHON" -m probeforge.cli evaluate --tasks data/tasks/test_ood.jsonl --backend no_probe --out runs/no_probe_ood --budgets 0 1 2 4 --no-traces
