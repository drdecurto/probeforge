"""Regenerate the checked-in Colab notebook without notebook dependencies."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})


md("""# ProbeForge — A100 pilot

Select **Runtime → Change runtime type → A100 GPU**. This notebook installs a
pinned text-model stack, trains Qwen3-8B with QLoRA, and saves data/checkpoints to
your Google Drive. It does not need a Nebius key. Full checkpoints consume Drive
space. Use a new `RUN_NAME` for a fresh experiment.

Upload `probeforge_training_pipeline.zip` in the next cell. No model weights are
included in that ZIP. The later download cell obtains the public Qwen3-8B weights.
""")
code("""from google.colab import files, drive
from pathlib import Path
import os, json, zipfile, subprocess, sys

PROJECT = Path('/content/probeforge')
if not PROJECT.exists():
    uploaded = files.upload()
    archives = [name for name in uploaded if name.endswith('.zip')]
    if len(archives) != 1:
        raise ValueError('Upload exactly one ProbeForge ZIP')
    with zipfile.ZipFile(archives[0]) as z:
        for member in z.infolist():
            target = (Path('/content') / member.filename).resolve()
            if not target.is_relative_to(Path('/content')):
                raise ValueError('Unsafe archive path')
        z.extractall('/content')
if not (PROJECT / 'pyproject.toml').exists():
    raise ValueError('Expected probeforge/pyproject.toml in archive')
os.chdir(PROJECT)
drive.mount('/content/drive')
DURABLE = Path('/content/drive/MyDrive/ProbeForge')
DURABLE.mkdir(parents=True, exist_ok=True)
RUN_NAME = 'a100-pilot-v1'
RUN_DIR = DURABLE / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
print('Project:', PROJECT, 'Run:', RUN_DIR)
""")
md("""## Install

Optional vision/audio packages are removed to avoid incompatibility with the pinned
Torch version. If Colab requests a runtime restart, restart and rerun the setup
cells. This changes only this Colab runtime.
""")
code("""subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[train,quant]'], check=True)
""")
code("""import torch
assert torch.cuda.is_available(), 'Select a GPU runtime'
gpu = torch.cuda.get_device_name(0)
print(gpu, round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')
assert 'A100' in gpu, 'This notebook targets A100; review memory settings for another GPU'

def run(*args):
    subprocess.run([sys.executable, '-m', 'probeforge.cli', *map(str, args)], check=True)

run('doctor')
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
""")
md("""## Data and immutable model snapshot

300 groups are a pilot. The same saved dataset is reused after a disconnect.
The first model download records a revision on Drive; subsequent downloads reuse
that exact revision. The local weights may disappear when the runtime resets.
""")
code("""DATA = RUN_DIR / 'data'
if not (DATA / 'tasks/manifest.json').exists():
    run('generate', '--out', DATA / 'tasks', '--groups', 300, '--seed', 42)
run('audit', '--data', DATA / 'tasks')
for split in ['train', 'validation']:
    dest = DATA / ('expert_' + split)
    if not (dest / 'manifest.json').exists():
        run('build', '--tasks', DATA / 'tasks' / (split + '.jsonl'), '--out', dest)

MODEL = Path('/content/models/Qwen3-8B')
revision_file = RUN_DIR / 'base_revision.json'
revision = json.loads(revision_file.read_text())['commit'] if revision_file.exists() else 'main'
if not (MODEL / 'download_manifest.json').exists():
    run('download', '--model', 'Qwen/Qwen3-8B', '--revision', revision, '--out', MODEL)
download = json.loads((MODEL / 'download_manifest.json').read_text())
if revision_file.exists():
    assert download['commit'] == json.loads(revision_file.read_text())['commit']
else:
    revision_file.write_text(json.dumps(download, indent=2))
""")
code("""cfg = json.loads((PROJECT / 'configs/sft_colab.json').read_text())
cfg.update(model=str(MODEL), train_data=str(DATA / 'expert_train/sft.jsonl'),
           validation_data=str(DATA / 'expert_validation/sft.jsonl'),
           output_dir=str(RUN_DIR / 'sft'), save_steps=10, eval_steps=10)
CONFIG = RUN_DIR / 'sft_config.json'
CONFIG.write_text(json.dumps(cfg, indent=2))
run('preflight', '--config', CONFIG)
""")
md("""## Train

Begin with 30 optimizer steps. For a continued smoke run, increase MAX_STEPS and
enable RESUME. For a clean full experiment, use a new RUN_NAME and MAX_STEPS=None.
Resuming a run whose short schedule already finished changes the learning-rate
schedule; use a fresh full run for the main experiment.
""")
code("""MAX_STEPS = 30
RESUME = False
args = ['train', '--config', CONFIG]
if MAX_STEPS is not None:
    args += ['--max-steps', MAX_STEPS]
if RESUME:
    args += ['--resume', 'latest']
run(*args)
print((RUN_DIR / 'sft/metrics.json').read_text())
""")
md("""## Small validation comparison

Use validation for development. Freeze settings before evaluating test_id and
test_ood. The two runs use the same quantization, tools, budgets, and selected groups.
Output directories must be fresh. Tiny pilot scores are not publication results.
""")
code("""for name, adapter in [('base_validation', None), ('sft_validation', RUN_DIR / 'sft')]:
    args = ['evaluate', '--tasks', DATA / 'tasks/validation.jsonl', '--backend', 'hf',
            '--model', MODEL, '--quantize', '--out', RUN_DIR / name,
            '--budgets', 0, 1, 2, '--max-groups', 2]
    if adapter is not None:
        args += ['--adapter', adapter]
    run(*args)
run('compare', '--a', RUN_DIR / 'base_validation/episodes.jsonl',
    '--b', RUN_DIR / 'sft_validation/episodes.jsonl', '--out', RUN_DIR / 'comparison.json')
""")
md("""## Optional DPO pilot

Run only after SFT and its evaluation succeed. Reference probabilities use the
same 4-bit configuration and unchanged SFT adapter. Start with exact preferences;
the README shows how to add student branching later.
""")
code("""RUN_DPO = False
if RUN_DPO:
    dpo = json.loads((PROJECT / 'configs/dpo_colab.json').read_text())
    dpo.update(model=str(MODEL), init_adapter=str(RUN_DIR / 'sft'),
               train_data=str(DATA / 'reference_train.jsonl'),
               validation_data=str(DATA / 'reference_validation.jsonl'),
               output_dir=str(RUN_DIR / 'dpo'), save_steps=10, eval_steps=10)
    DPO_CONFIG = RUN_DIR / 'dpo_config.json'
    DPO_CONFIG.write_text(json.dumps(dpo, indent=2))
    for split in ['train', 'validation']:
        run('cache-reference', '--config', DPO_CONFIG,
            '--input', DATA / ('expert_' + split) / 'preferences.jsonl',
            '--out', DATA / ('reference_' + split + '.jsonl'))
    run('train', '--config', DPO_CONFIG, '--stage', 'dpo', '--max-steps', 30)
""")
md("""## Download the adapter

Drive contains optimizer checkpoints for resuming. The ZIP below contains only
the final adapter, tokenizer, and run metadata. Preserve the model revision file
too. Deploy with the original backbone plus this adapter; merge on a machine with
adequate host RAM if a standalone checkpoint is required.
""")
code("""import shutil, tempfile
ADAPTER = RUN_DIR / 'sft'  # change to 'dpo' if that checkpoint is preferable
with tempfile.TemporaryDirectory() as folder:
    dest = Path(folder) / 'Probe-8B-adapter'
    dest.mkdir()
    for item in ADAPTER.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
    shutil.copy2(revision_file, dest / 'base_revision.json')
    archive = shutil.make_archive('/content/Probe-8B-adapter', 'zip', folder)
files.download(archive)
""")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "ProbeForge A100 pilot", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"probeforge-{i:02d}"
path = Path(__file__).resolve().parents[1]/"notebooks/colab_a100.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, indent=2)+"\n")
