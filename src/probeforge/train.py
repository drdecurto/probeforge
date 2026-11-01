import importlib.metadata
import json
import os
import random
from pathlib import Path

from .io import read_jsonl, write_json, write_jsonl, file_hash, digest
from .training_data import encode_row, CompletionCollator, sequence_logps, row_fingerprint


def model_fingerprint(model, revision, adapter=None):
    result = {"model": model, "revision": revision}
    if Path(model).is_dir():
        # Record actual weight shards, not just a mutable directory name.
        paths = sorted(Path(model).glob("*.safetensors")) + sorted(Path(model).glob("pytorch_model*.bin"))
        result["weights"] = {p.name: file_hash(p) for p in paths}
        result["config"] = file_hash(Path(model)/"config.json")
    if adapter:
        root = Path(adapter)
        result["adapter"] = {p.name: file_hash(p) for p in sorted(root.glob("adapter*")) if p.is_file()}
        if not result["adapter"]:
            raise ValueError("No adapter files found")
        result["tokenizer"] = {p.name: file_hash(p) for p in sorted(root.glob("*token*")) if p.is_file()}
    return digest(result)


def reference_fingerprint(cfg):
    import torch
    if not Path(cfg["model"]).is_dir() and cfg.get("revision", "main") == "main":
        raise ValueError("Reference caching needs a local model snapshot or immutable revision")
    return digest({"weights_and_tokenizer": model_fingerprint(cfg["model"], cfg.get("revision", "main"), cfg["init_adapter"]),
                   "quantize": cfg.get("quantize", False),
                   "compute_dtype": "bfloat16" if torch.cuda.is_available() else "float32"})


def load_config(path, overrides=None):
    cfg = json.loads(Path(path).read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def load_model(cfg, trainable=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if cuda else "cpu"
    if not cuda and not cfg.get("allow_cpu", False):
        raise ValueError("Training requires CUDA. Use the CPU smoke test for correctness; Mac is for data and inference.")
    dtype = torch.bfloat16 if cuda else torch.float32
    base, adapter = cfg["model"], cfg.get("init_adapter")
    tokenizer = AutoTokenizer.from_pretrained(adapter or base, revision="main" if adapter else cfg.get("revision", "main"), trust_remote_code=False)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("Tokenizer needs an EOS token and a chat template")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"revision": cfg.get("revision", "main"), "torch_dtype": dtype,
              "attn_implementation": "sdpa", "trust_remote_code": False}
    if cfg.get("quantize", False):
        if not cuda:
            raise ValueError("QLoRA requires CUDA in this pipeline")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": local_rank}
    elif cfg.get("device_map"):
        # Naive model parallelism over the GPUs visible to this process (for models larger than one GPU).
        kwargs["device_map"] = cfg["device_map"]
        if cfg.get("max_memory_gib"):
            kwargs["max_memory"] = {i: f"{cfg['max_memory_gib']}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if cfg.get("quantize", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=trainable)
    elif not cfg.get("device_map"):
        model.to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
                   lora_dropout=cfg.get("lora_dropout", .05), bias="none", task_type="CAUSAL_LM",
                   target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])))
    model.config.use_cache = False
    if trainable:
        model.enable_input_require_grads()
    return model, tokenizer, device


def validate_splits(train_rows, validation_rows):
    if not train_rows or not validation_rows:
        raise ValueError("Both train and validation must be nonempty")
    if any(r.get("split") != "train" for r in train_rows):
        raise ValueError("Training rows must have split=train")
    if any(r.get("split") != "validation" for r in validation_rows):
        raise ValueError("Validation rows must have split=validation")
    overlap = {r["group_id"] for r in train_rows} & {r["group_id"] for r in validation_rows}
    if overlap:
        raise ValueError("Group leakage between train and validation")


def train(cfg, stage="sft", resume=None):
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    def _ta(**kw):
        # transformers 5 removed some arguments (e.g. warmup_ratio -> float warmup_steps); keep 4.x behaviour unchanged.
        import inspect
        ok = set(inspect.signature(TrainingArguments.__init__).parameters)
        if "warmup_ratio" in kw and "warmup_ratio" not in ok:
            kw["warmup_steps"] = kw.pop("warmup_ratio")
        drop = [k for k in kw if k not in ok]
        if drop:
            print("TrainingArguments: dropping unsupported", drop, flush=True)
        return TrainingArguments(**{k: v for k, v in kw.items() if k in ok})
    root = Path(cfg["output_dir"])
    rank = int(os.environ.get("RANK", 0))
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("Output directory is nonempty; use --resume or choose another directory")
    if resume and (root/"run_manifest.json").exists():
        previous = json.loads((root/"run_manifest.json").read_text())
        for label, key in (("train_sha256", "train_data"), ("validation_sha256", "validation_data")):
            if previous[label] != file_hash(cfg[key]):
                raise ValueError("Resume dataset differs from the checkpoint run")
        for key in ("model", "revision", "init_adapter", "quantize", "seed", "max_length",
                    "lora_rank", "lora_alpha", "target_modules", "batch_size", "gradient_accumulation"):
            if previous["config"].get(key) != cfg.get(key):
                raise ValueError(f"Resume changed protected setting: {key}")
    set_seed(cfg.get("seed", 42))
    train_rows = list(read_jsonl(cfg["train_data"]))
    validation_rows = list(read_jsonl(cfg["validation_data"]))
    validate_splits(train_rows, validation_rows)
    model, tokenizer, device = load_model(cfg)
    if stage == "dpo":
        if not cfg.get("init_adapter"):
            raise ValueError("DPO requires init_adapter pointing to the SFT checkpoint")
        fp = reference_fingerprint(cfg)
        for row in train_rows + validation_rows:
            if row.get("reference_fingerprint") != fp or row.get("reference_row") != row_fingerprint(row):
                raise ValueError("Stale/missing DPO reference values; run cache-reference with this exact model and SFT adapter")
            if row.get("reference_max_length") != cfg["max_length"]:
                raise ValueError("Reference max_length differs")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    encoded_train = [encode_row(tokenizer, r, cfg["max_length"], stage) for r in train_rows]
    encoded_val = [encode_row(tokenizer, r, cfg["max_length"], stage) for r in validation_rows]
    collator = CompletionCollator(tokenizer.pad_token_id, stage)

    class DPOTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logps = {}
            for side in ("chosen", "rejected"):
                output = model(input_ids=inputs[side+"_input_ids"], attention_mask=inputs[side+"_attention_mask"])
                logps[side] = sequence_logps(output.logits, inputs[side+"_labels"])
            delta = (logps["chosen"]-logps["rejected"])-(inputs["ref_chosen"]-inputs["ref_rejected"])
            loss = -torch.nn.functional.logsigmoid(cfg.get("beta", .1)*delta).mean()
            return (loss, {"logits": delta.detach().unsqueeze(-1)}) if return_outputs else loss

    args = _ta(output_dir=str(root), num_train_epochs=cfg.get("epochs", 1),
                max_steps=cfg.get("max_steps", -1), learning_rate=cfg.get("learning_rate", 1e-4),
                per_device_train_batch_size=cfg.get("batch_size", 1), per_device_eval_batch_size=1,
                gradient_accumulation_steps=cfg.get("gradient_accumulation", 8),
                gradient_checkpointing=cfg.get("gradient_checkpointing", True),
                gradient_checkpointing_kwargs={"use_reentrant": False},
                bf16=device.startswith("cuda"), fp16=False, tf32=device.startswith("cuda"),
                optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=.05, max_grad_norm=1.0,
                logging_steps=cfg.get("logging_steps", 10), eval_strategy="steps", save_strategy="steps",
                eval_steps=cfg.get("eval_steps", 100), save_steps=cfg.get("save_steps", 100), save_total_limit=2,
                report_to="none", remove_unused_columns=stage == "sft", prediction_loss_only=True,
                ddp_find_unused_parameters=False, dataloader_num_workers=0, seed=cfg.get("seed", 42),
                use_cpu=not device.startswith("cuda"), label_names=["labels"] if stage == "sft" else ["chosen_labels", "rejected_labels"],
                save_safetensors=True)
    cls = Trainer if stage == "sft" else DPOTrainer
    trainer = cls(model=model, args=args, train_dataset=encoded_train, eval_dataset=encoded_val,
                  data_collator=collator, processing_class=tokenizer)
    # Custom DPO loss is an example mean; let Trainer scale gradient accumulation.
    if stage == "dpo":
        trainer.model_accepts_loss_kwargs = False
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}
        write_json(root / "run_manifest.json", {"config": cfg, "stage": stage, "versions": versions,
                   "train_sha256": file_hash(cfg["train_data"]), "validation_sha256": file_hash(cfg["validation_data"]),
                   "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                   "effective_batch": cfg.get("batch_size", 1)*cfg.get("gradient_accumulation", 8)*int(os.environ.get("WORLD_SIZE", 1)),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "model_commit": getattr(model.config, "_commit_hash", None)})
    checkpoint = get_last_checkpoint(str(root)) if resume == "latest" else resume
    if resume == "latest" and not checkpoint:
        raise ValueError("No checkpoint found for --resume latest")
    result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = trainer.evaluate() if cfg.get("final_eval", True) else {}
    trainer.save_model(str(root))
    trainer.save_state()
    if rank == 0:
        tokenizer.save_pretrained(root)
        write_json(root / "metrics.json", {**result.metrics, **metrics})
    return {**result.metrics, **metrics}


def cache_reference(cfg, input_path, output_path):
    import torch
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Reference caching runs on one GPU; do not torchrun")
    if not cfg.get("init_adapter"):
        raise ValueError("Set init_adapter to the SFT checkpoint")
    model, tokenizer, device = load_model(cfg, trainable=False)
    model.eval()
    fp = reference_fingerprint(cfg)
    rows = []
    collator = CompletionCollator(tokenizer.pad_token_id, "dpo")
    with torch.inference_mode():
        for row in read_jsonl(input_path):
            if row["split"] not in ("train", "validation"):
                raise ValueError("Refuse evaluation splits in training reference cache")
            encoded = encode_row(tokenizer, row, cfg["max_length"], "dpo")
            batch = {k: v.to(device) for k, v in collator([encoded]).items()}
            row = dict(row)
            for side in ("chosen", "rejected"):
                output = model(input_ids=batch[side+"_input_ids"], attention_mask=batch[side+"_attention_mask"])
                row["ref_"+side] = sequence_logps(output.logits, batch[side+"_labels"]).item()
            row.update(reference_fingerprint=fp, reference_row=row_fingerprint(row), reference_max_length=cfg["max_length"])
            rows.append(row)
    write_jsonl(output_path, rows)
    return {"rows": len(rows), "reference_fingerprint": fp}


def export_model(model_path, adapter, out, revision="main"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    if Path(out).exists():
        raise FileExistsError("Export destination exists")
    base = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, torch_dtype=torch.bfloat16, trust_remote_code=False)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    write_json(Path(out)/"probeforge_provenance.json", {"base": model_path, "revision": revision,
                                                       "adapter_fingerprint": model_fingerprint(model_path, revision, adapter)})
