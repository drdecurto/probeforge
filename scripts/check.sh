#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m unittest discover -s tests -v
python -m compileall -q src scripts
for file in scripts/*.sh slurm/*.sh slurm/*.sbatch; do bash -n "$file"; done
