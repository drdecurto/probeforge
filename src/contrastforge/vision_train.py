"""Optional Qwen2.5-VL SFT with a frozen visual encoder and text-attention LoRA.

One image per batch. No token truncation. Requires transformers==4.51.3.
"""
import os
from pathlib import Path
from probeforge.io import read_jsonl,write_json,file_hash,canonical
from .vision import image_for
from .dataset import gold


def load_vision(cfg,trainable=True):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
    from peft import PeftModel,LoraConfig,get_peft_model,prepare_model_for_kbit_training
    cuda=torch.cuda.is_available();device=f"cuda:{int(os.environ.get('LOCAL_RANK',0))}" if cuda else "cpu"
    if not cuda and not cfg.get("allow_cpu"):raise ValueError("VLM training requires CUDA")
    dtype=torch.bfloat16 if cuda else torch.float32
    kwargs={"torch_dtype":dtype,"revision":cfg.get("revision","main"),"attn_implementation":"sdpa","trust_remote_code":False}
    if cfg.get("quantize"):
        if not cuda:raise ValueError("Quantized VLM path requires CUDA")
        kwargs.update(device_map={"":int(os.environ.get("LOCAL_RANK",0))},
          quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["model"],**kwargs)
    processor=AutoProcessor.from_pretrained(cfg.get("init_adapter") or cfg["model"],trust_remote_code=False,
                 min_pixels=56*56,max_pixels=cfg.get("max_pixels",256*28*28))
    if cfg.get("quantize"):model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=trainable)
    else:model.to(device)
    if cfg.get("init_adapter"):model=PeftModel.from_pretrained(model,cfg["init_adapter"],is_trainable=trainable)
    elif trainable:
        names=[name for name,module in model.named_modules() if ".self_attn." in name and
               name.rsplit(".",1)[-1] in ("q_proj","k_proj","v_proj","o_proj") and "visual" not in name]
        if not names:raise ValueError("No supported text attention modules found")
        model=get_peft_model(model,LoraConfig(r=cfg.get("lora_rank",16),lora_alpha=cfg.get("lora_alpha",32),
                    lora_dropout=.05,target_modules=names,task_type="CAUSAL_LM",bias="none"))
    if trainable:
        if any(p.requires_grad for n,p in model.named_parameters() if "visual" in n):
            raise ValueError("Visual encoder must remain frozen")
        model.enable_input_require_grads()
    model.config.use_cache=not trainable
    return model,processor,device


def vision_inputs(processor,row,completion=None):
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["prompt"][0]["content"]}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    image=image_for(row)
    prefix=processor(text=[text],images=[image],padding=False,return_tensors="pt")
    if completion is None:return prefix
    full=processor(text=[text+completion+processor.tokenizer.eos_token],images=[image],padding=False,return_tensors="pt")
    n=prefix["input_ids"].shape[1]
    if not full["input_ids"][0,:n].equal(prefix["input_ids"][0]):raise ValueError("Prompt boundary changed during tokenization")
    full["labels"]=full["input_ids"].clone();full["labels"][:,:n]=-100
    return full


def train_vision(cfg,resume=None):
    from transformers import Trainer,TrainingArguments,set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from .dataset import audit_rows
    if int(os.environ.get("WORLD_SIZE",1))!=1:raise ValueError("VLM path supports one GPU; run independent jobs")
    root=Path(cfg["output_dir"])
    if root.exists() and any(root.iterdir()) and not resume:raise FileExistsError("Output exists; use --resume")
    rows={split:list(read_jsonl(cfg[key])) for split,key in (("train","train_data"),("validation","validation_data"))}
    for split,rr in rows.items():
        if not rr or any(r["split"]!=split or r["kind"]!="vlm" for r in rr):raise ValueError("Wrong vision training split")
    audit_rows(rows["train"]+rows["validation"])
    set_seed(cfg.get("seed",42));model,processor,device=load_vision(cfg)
    class Collator:
        def __call__(self,batch):
            if len(batch)!=1:raise ValueError("Vision collator requires batch_size=1")
            data=vision_inputs(processor,batch[0],canonical(gold(batch[0])))
            if data["input_ids"].shape[1]>cfg["max_length"]:raise ValueError("VLM example exceeds max_length; reduce max_pixels or increase context")
            return data
    args=TrainingArguments(output_dir=str(root),num_train_epochs=cfg.get("epochs",1),max_steps=cfg.get("max_steps",-1),
       learning_rate=cfg.get("learning_rate",1e-4),per_device_train_batch_size=1,per_device_eval_batch_size=1,
       gradient_accumulation_steps=cfg.get("gradient_accumulation",16),gradient_checkpointing=True,
       gradient_checkpointing_kwargs={"use_reentrant":False},bf16=device.startswith("cuda"),
       use_cpu=not device.startswith("cuda"),optim="adamw_torch",lr_scheduler_type="cosine",warmup_ratio=.05,
       eval_strategy="steps",save_strategy="steps",eval_steps=cfg.get("eval_steps",50),save_steps=cfg.get("save_steps",50),
       logging_steps=5,save_total_limit=2,report_to="none",remove_unused_columns=False,label_names=["labels"],
       prediction_loss_only=True,dataloader_num_workers=0,seed=cfg.get("seed",42))
    trainer=Trainer(model=model,args=args,train_dataset=rows["train"],eval_dataset=rows["validation"],
                    data_collator=Collator(),processing_class=processor)
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"run_manifest.json",{"config":cfg,"frozen_visual_encoder":True,
               "train_hash":file_hash(cfg["train_data"]),"validation_hash":file_hash(cfg["validation_data"])})
    checkpoint=get_last_checkpoint(str(root)) if resume=="latest" else resume
    if resume=="latest" and not checkpoint:raise ValueError("No checkpoint to resume")
    result=trainer.train(resume_from_checkpoint=checkpoint);metrics={**result.metrics,**trainer.evaluate()}
    trainer.save_model(str(root));processor.save_pretrained(root);trainer.save_state()
    write_json(root/"metrics.json",metrics);return metrics
