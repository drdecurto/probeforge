# MareNostrum 5 operating guide

## Allocation and environment

This path assumes access to the accelerated partition. Published MN5 descriptions
specify four H100 GPUs with 64 GB per accelerated node. An allocation to general
purpose CPU nodes alone does not provide this GPU path.

Use the actual account, QoS, and optional partition assigned to your project. This
repository does not hardcode guessed `acc_*` names. On BSC, inspect the allocation
instructions and permitted scheduler information, for example:

```bash
sinfo -o '%P %G %l'
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS
```

Some sites restrict `sacctmgr`; your project documentation remains authoritative.
The account/QoS are the only scheduler values that must be supplied to submit.sh.
The helper uses standard `--gres=gpu:N`. Adjust that resource flag if your BSC
allocation instructions require a different GPU resource expression.

Prepare a Python 3.11/3.12 environment against BSC-supported CUDA/driver modules.
The supplied Torch 2.6.0 Linux package is a CUDA 12.4-era build. Validate import,
`probeforge doctor`, and a short training run on an allocated GPU. Do not launch
model training on a login node. If your project mandates Singularity/Apptainer,
build the environment inside the approved container and replace the Python launch
in job.sbatch with that container's documented invocation; do not transplant a
host venv into an incompatible container. Module/container names are site-specific.

Use project storage for data, model snapshots, and checkpoints. Download packages
and model files while network access is available. Compute jobs default to
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. They never install dependencies.

```bash
export PF_ACCOUNT='YOUR_ACCOUNT'
export PF_QOS='YOUR_ACCELERATED_QOS'
export PF_PYTHON="$PWD/.venv/bin/python"
export PF_TIME=04:00:00
# Optional:
# export PF_PARTITION='YOUR_PARTITION'
# export PF_ENV_SETUP="$PWD/slurm/environment.local.sh"
```

`PF_ENV_SETUP` is sourced inside the job to load your modules. Start from
`slurm/environment.example.sh`. `PF_PYTHON` must be an absolute executable path.

## Main training

```bash
bash slurm/submit.sh sft configs/sft_mn5.json --max-steps 30 --output-dir runs/mn5_pilot
bash slurm/submit.sh sft configs/sft_mn5.json
```

Do not launch both into the same output path. The first command returns a Slurm
job ID; inspect it before starting a longer allocation:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS
```

The launcher reserves one node and one Slurm task. That task runs torchrun, which
starts four workers. Do not wrap it in a four-task `srun`, which would create 16
training processes. Each worker gets its own backbone and LoRA adapter.

For a time-limited job that has a saved checkpoint:

```bash
bash slurm/submit.sh sft configs/sft_mn5.json --resume latest
```

Resume loads the optimizer, scheduler, and RNG state from a checkpoint. The final
adapter alone is not a resumable optimizer checkpoint. At most two periodic
checkpoints are retained. There is no guaranteed save at scheduler termination;
set a sensible save interval before a long run.

## Reference probabilities and DPO

Reference jobs are single-process GPU jobs. If your QoS allocates whole nodes only,
set `PF_GPUS=4` for them; reference caching itself uses GPU 0. Two independent
reference jobs can be submitted with different output files:

```bash
bash slurm/submit.sh reference configs/dpo_mn5.json \
  --input data/expert_train/preferences.jsonl --out data/reference_train.jsonl
bash slurm/submit.sh reference configs/dpo_mn5.json \
  --input data/expert_validation/preferences.jsonl --out data/reference_validation.jsonl
```

After both complete successfully:

```bash
bash slurm/submit.sh dpo configs/dpo_mn5.json
```

Use `data/preferences_round1.jsonl` instead of expert preferences for the student
branch experiment. If the SFT adapter, base model, or token limit changes, regenerate
both reference caches. Do not overwrite an adapter while a cache job is reading it.
Optional job dependency: set `PF_DEPENDENCY='JOB1:JOB2'` before submission; the helper
uses `afterok`. Unset it afterward if it should not affect later jobs.

## Local student collection and evaluation

These commands do not need Nebius or outbound network:

```bash
bash slurm/submit.sh collect - --tasks data/tasks/train.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --out data/student_round1 --limit 200

bash slurm/submit.sh branch - --tasks data/tasks/train.jsonl \
  --episodes data/student_round1/episodes.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --out data/branches_round1 --max-states 100 --worlds 4 --repeats 1

bash slurm/submit.sh evaluate - --tasks data/tasks/test_ood.jsonl --backend hf \
  --model models/Qwen3-8B --adapter runs/probe8b_sft_seed42 \
  --out runs/sft_ood --budgets 0 1 2 4
```

Run the corresponding base evaluation with the same arguments and without
`--adapter`. Use separate output directories. Model inference is sequential in
this first implementation; measure rollout time before collecting thousands of
episodes. The API backend can also target a separately managed vLLM server, but
the repository does not start an inference service automatically.

## Multiple seeds and resource accounting

```bash
for seed in 42 43 44; do
  bash slurm/submit.sh sft configs/sft_mn5.json --seed "$seed" \
    --output-dir "runs/probe8b_sft_seed$seed"
done
```

Use the same data split for optimization-seed comparisons; vary data-generation
seeds as a separate robustness experiment. Report allocated GPU-hours = allocated
GPU count × elapsed hours, including idle GPUs when whole-node allocation is
required. Distinguish training, reference caching, collection, branching, and final
evaluation. The software does not promise a wall-clock completion time.

Sources: [MN5 system paper](https://arxiv.org/abs/2503.09917),
[Slurm GPU resources](https://slurm.schedmd.com/gres.html),
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
