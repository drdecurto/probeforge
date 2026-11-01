import argparse
import importlib.util
import json
import os
import platform
import sys
import urllib.request
from pathlib import Path

from .io import read_jsonl, write_json


def policy_args(p):
    p.add_argument("--backend", choices=["oracle", "no_probe", "random", "hf", "api"], default="oracle")
    p.add_argument("--model")
    p.add_argument("--adapter")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="auto")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--base-url")
    p.add_argument("--key-env", default="NEBIUS_API_KEY")
    p.add_argument("--cache", default="data/api-cache")
    p.add_argument("--max-requests", type=int, default=1000)
    p.add_argument("--extra-json", default="{}", help="Optional provider-specific request fields")


def get_policy(args):
    from .policies import RulePolicy, APIPolicy, HFPolicy
    if args.backend in ("oracle", "no_probe", "random"):
        return RulePolicy(args.backend)
    if not args.model:
        raise ValueError("--model is required for hf/api")
    if args.backend == "api":
        return APIPolicy(args.model, args.base_url, args.key_env, args.cache, args.max_requests,
                         args.max_tokens, args.temperature, json.loads(args.extra_json))
    return HFPolicy(args.model, args.adapter, args.device, args.quantize, args.max_tokens, args.temperature, args.revision)


def config_args(p):
    p.add_argument("--config", required=True)
    for name in ("model", "revision", "init-adapter", "train-data", "validation-data", "output-dir"):
        p.add_argument("--"+name)
    for name in ("max-steps", "seed", "max-length", "gradient-accumulation"):
        p.add_argument("--"+name, type=int)


def cfg_from(args):
    from .train import load_config
    keys = ("model", "revision", "init_adapter", "train_data", "validation_data", "output_dir",
            "max_steps", "seed", "max_length", "gradient_accumulation")
    return load_config(args.config, {k: getattr(args, k, None) for k in keys})


def record_arguments(args):
    if not getattr(args, "out", None) or args.command not in ("collect", "branch", "evaluate"):
        return
    path = Path(args.out)/"arguments.json"
    values = vars(args).copy()
    if path.exists() and getattr(args, "resume", False):
        prior = json.loads(path.read_text())
        mutable = {"resume", "limit", "max_requests"}
        if {k: v for k, v in prior.items() if k not in mutable} != {k: v for k, v in values.items() if k not in mutable}:
            raise ValueError("Resume policy/config differs; choose a new collection directory")
    elif path.exists():
        raise FileExistsError("Output arguments already exist; choose a new directory")
    write_json(path, values)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ProbeForge executable training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--out", required=True); p.add_argument("--groups", type=int, default=300); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("build")
    p.add_argument("--tasks", required=True); p.add_argument("--out", required=True); p.add_argument("--cost-penalty", type=float, default=.02)
    p = sub.add_parser("collect"); policy_args(p)
    p.add_argument("--tasks", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--resume", action="store_true"); p.add_argument("--source", default="student")
    p = sub.add_parser("branch"); policy_args(p)
    p.add_argument("--tasks", required=True); p.add_argument("--episodes", required=True); p.add_argument("--out", required=True)
    for name, default in (("max-states", 100), ("worlds", 4), ("repeats", 2), ("max-actions", 4), ("seed", 42)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--cost-penalty", type=float, default=.02)
    p = sub.add_parser("mix"); p.add_argument("--inputs", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--cap-per-source", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("evaluate"); policy_args(p)
    p.add_argument("--tasks", required=True); p.add_argument("--out", required=True); p.add_argument("--budgets", nargs="+", type=int, default=[0, 1, 2, 4])
    p.add_argument("--max-groups", type=int, default=0); p.add_argument("--seed", type=int, default=42); p.add_argument("--no-traces", action="store_true")
    p.add_argument("--intervention", choices=["none", "random_probe"], default="none")
    p = sub.add_parser("compare"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("train"); config_args(p); p.add_argument("--stage", choices=["sft", "dpo"], default="sft"); p.add_argument("--resume", nargs="?", const="latest")
    p = sub.add_parser("cache-reference"); config_args(p); p.add_argument("--input", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("export"); p.add_argument("--model", required=True); p.add_argument("--adapter", required=True); p.add_argument("--out", required=True); p.add_argument("--revision", default="main")
    p = sub.add_parser("download"); p.add_argument("--model", default="Qwen/Qwen3-8B"); p.add_argument("--revision", default="main"); p.add_argument("--out", required=True)
    p = sub.add_parser("preflight"); config_args(p); p.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    p = sub.add_parser("audit"); p.add_argument("--data", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("models"); p.add_argument("--base-url"); p.add_argument("--key-env", default="NEBIUS_API_KEY")
    args = parser.parse_args(argv)
    try:
        record_arguments(args)
        cmd = args.command
        if cmd == "generate":
            from .tasks import generate_dataset
            result = generate_dataset(args.out, args.groups, args.seed)
        elif cmd == "build":
            from .data import make_expert_data
            result = make_expert_data(args.tasks, args.out, args.cost_penalty)
        elif cmd == "collect":
            from .data import collect
            result = collect(args.tasks, args.out, get_policy(args), args.limit, args.seed, args.resume, args.source)
        elif cmd == "branch":
            from .data import branch_preferences
            if min(args.max_states, args.worlds, args.repeats, args.max_actions) < 1:
                raise ValueError("Branch limits must be positive")
            result = branch_preferences(args.tasks, args.episodes, args.out, get_policy(args), args.max_states,
                                         args.worlds, args.repeats, args.max_actions, args.cost_penalty, args.seed)
        elif cmd == "mix":
            from .data import mix
            result = mix(args.inputs, args.out, args.seed, args.cap_per_source)
        elif cmd == "evaluate":
            from .evaluation import evaluate
            if any(b < 0 or b > 6 for b in args.budgets) or len(set(args.budgets)) != len(args.budgets):
                raise ValueError("Use distinct budgets from 0 through 6")
            policy = get_policy(args)
            if args.intervention == "random_probe":
                from .policies import RandomProbeIntervention
                policy = RandomProbeIntervention(policy)
            result = evaluate(args.tasks, args.out, policy, args.budgets, args.max_groups, args.seed, not args.no_traces)
        elif cmd == "compare":
            from .evaluation import compare
            result = compare(args.a, args.b, args.out)
        elif cmd == "train":
            from .train import train
            result = train(cfg_from(args), args.stage, args.resume)
        elif cmd == "cache-reference":
            from .train import cache_reference
            result = cache_reference(cfg_from(args), args.input, args.out)
        elif cmd == "export":
            from .train import export_model
            export_model(args.model, args.adapter, args.out, args.revision)
            result = {"exported": args.out}
        elif cmd == "download":
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().model_info(args.model, revision=args.revision)
            snapshot_download(args.model, revision=info.sha, local_dir=args.out,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md", "LICENSE*"])
            result = {"model": args.model, "commit": info.sha, "path": str(Path(args.out).resolve())}
            write_json(Path(args.out)/"download_manifest.json", result)
        elif cmd == "preflight":
            from transformers import AutoTokenizer
            from .training_data import encode_row
            from .train import validate_splits
            cfg = cfg_from(args)
            train_rows, val_rows = list(read_jsonl(cfg["train_data"])), list(read_jsonl(cfg["validation_data"]))
            validate_splits(train_rows, val_rows)
            tok = AutoTokenizer.from_pretrained(cfg.get("init_adapter") or cfg["model"],
                                               revision="main" if cfg.get("init_adapter") else cfg.get("revision", "main"),
                                               trust_remote_code=False)
            lengths = []
            for row in train_rows + val_rows:
                enc = encode_row(tok, row, cfg["max_length"], args.stage)
                lengths.append(len(enc["input_ids"] if args.stage == "sft" else enc["chosen_input_ids"]))
            result = {"train_rows": len(train_rows), "validation_rows": len(val_rows), "max_tokens": max(lengths), "mean_tokens": sum(lengths)/len(lengths)}
        elif cmd == "audit":
            from .validation import audit_dataset
            result = audit_dataset(args.data)
        elif cmd == "doctor":
            result = {"python": sys.version, "platform": platform.platform(), "slurm": bool(os.getenv("SLURM_JOB_ID")),
                      "nebius_key_present": bool(os.getenv("NEBIUS_API_KEY"))}
            if importlib.util.find_spec("torch"):
                import torch
                result.update(torch=torch.__version__, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available(),
                              gpu_count=torch.cuda.device_count(), gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        elif cmd == "models":
            url = (args.base_url or os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")).rstrip("/")
            key = os.environ.get(args.key_env)
            if not key:
                raise ValueError(f"Set {args.key_env}")
            req = urllib.request.Request(url+"/models", headers={"Authorization": "Bearer "+key})
            with urllib.request.urlopen(req, timeout=30) as response:
                result = {"models": [r["id"] for r in json.load(response)["data"]]}
        print(json.dumps(result, indent=2))
    except (ValueError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
