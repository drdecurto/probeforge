# Mac, Colab, and MN5

## Mac M3 with 32 GB

Use the Mac for the standard-library environment generator, data validation,
Nebius calls, report analysis, and lightweight inference. No CUDA or paid endpoint
is needed to prepare the exact-teacher dataset.

For a small local inference experiment, install the training dependencies into a
native Apple Silicon Python 3.11/3.12 environment and choose a small checkpoint:

```bash
python -m pip install -e '.[train]'
probeforge download --model Qwen/Qwen3-0.6B --out models/Qwen3-0.6B
probeforge evaluate --tasks data/tasks/validation.jsonl --backend hf \
  --model models/Qwen3-0.6B --device mps --budgets 1 2 --max-groups 2 \
  --out runs/mac_small_validation
```

The HF backend uses FP16 on MPS. Do not pass `--quantize`: this repository's 4-bit
path uses bitsandbytes on NVIDIA CUDA. The tiny CPU smoke test also runs on a Mac
with training dependencies, although hardware-specific execution was not tested
when packaging the repository. An 8B MPS inference attempt may fit at short context,
but 32 GB is shared with macOS and other applications; start smaller. Full 8B
training on this Mac is not the supported/recommended route. No MLX training path
is included or implied.

## Colab A100

Select an A100 GPU runtime explicitly; paid Colab plans do not guarantee that a
particular accelerator is available. Confirm its device name and memory before
training. Use the notebook. The QLoRA config targets one A100, including 40 GB
variants, with batch size 1 and 2,048-token context. Actual peak memory must be
measured in the pilot. If it does not fit, lower max_length only after checking
data lengths; do not silently truncate examples.

Colab preinstalls changing packages. The notebook installs a pinned text-model
stack and removes optional torchvision/torchaudio packages that can otherwise
conflict with the selected Torch build. Restart the runtime if required after
installation, then rerun setup cells. Checkpoints and datasets go to your Drive;
local model files under /content may need to be restored after disconnecting.

Keep the data, seeds, tokenizer, context limit, and effective batch identical when
moving the pilot to MN5. BF16 LoRA and NF4 QLoRA still differ numerically. For a
closer comparison, use the same quantization setting on both systems and log it.

## MareNostrum

Main path: one accelerated node, four local workers, BF16 LoRA. Four 64 GB GPUs
are four independently limited memory spaces in this setup. Gradient checkpointing
reduces activation storage; frozen weights still participate in the forward and
backward computations. LoRA reduces optimizer memory far more than it reduces
backbone computation.

Preflight checks token lengths and splits without loading the full model. Then use
a 30-step allocated pilot to measure actual training speed and memory. Estimate
the full run from processed examples/tokens, not adapter file size. Include all
reference/rollout/evaluation work in the compute budget.

## Package reproducibility

Core tested pins: Torch 2.6.0, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0.
The release's CPU verification uses the 2.6.0+cpu wheel. GPU wheels and platform
drivers must be checked on the actual machine. `run_manifest.json` records installed
core versions and effective batch. Save `python -m pip freeze` after a successful
GPU pilot if you want a complete machine-specific lock file.

Official sources:
[Qwen3 model support](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen3),
[Qwen3-8B card](https://huggingface.co/Qwen/Qwen3-8B),
[PEFT LoRA guide](https://huggingface.co/docs/peft/en/developer_guides/lora),
[Nebius quickstart](https://docs.tokenfactory.nebius.com/quickstart).
