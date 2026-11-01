"""Real CPU SFT/resume/DPO/export check with tiny random Qwen3 and the NEW curriculum."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
from safetensors.torch import load_file
from probeforge.io import read_jsonl,write_jsonl,write_json
from probeforge.train import train,cache_reference,export_model
from contrastforge.dataset import generate
from contrastforge.collect import collect
from contrastforge.curriculum import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);args=p.parse_args()
    root=Path(args.out).resolve()
    if root.exists():raise FileExistsError("Use a fresh smoke directory")
    torch.set_num_threads(2);torch.manual_seed(42)
    generate(root/"raw",6,languages=["en","es","ar"])
    inputs=[root/"raw/train.jsonl",root/"raw/validation.jsonl"]
    collect(inputs,root/"fixture.jsonl",backend="mock")
    prepare(inputs,root/"curr",rollouts=[root/"fixture.jsonl"],allow_synthetic=True)
    texts=[]
    for split in ("train","validation"):
        rows=list(read_jsonl(root/f"curr/sft_{split}.jsonl"))[:8]
        write_jsonl(root/f"sft_{split}.jsonl",rows)
        write_jsonl(root/f"prefs_{split}.jsonl",list(read_jsonl(root/f"curr/preferences_{split}.jsonl"))[:4])
        texts.extend(" ".join(m["content"] for m in r["prompt"])+r["completion"] for r in rows)
    tok=Tokenizer(models.BPE(unk_token="<unk>"));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator(texts,trainers.BpeTrainer(vocab_size=1000,special_tokens=["<pad>","<unk>","<eos>"]))
    template="{% for m in messages %}{{m['role'] + ':\n' + m['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'assistant:\n'}}{% endif %}"
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token="<pad>",unk_token="<unk>",eos_token="<eos>",chat_template=template)
    base=root/"tiny-base";tokenizer.save_pretrained(base)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=64,
       num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=2048,
       eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,tie_word_embeddings=True))
    model.save_pretrained(base)
    # Check the token-budget comparison mechanism with complete groups.
    for strategy in ("translated","mined"):
        report=prepare(inputs,root/f"budget_{strategy}",strategy=strategy,rollouts=[root/"fixture.jsonl"],
                       allow_synthetic=True,token_budget=16000,tokenizer_path=str(base))
        assert 0<report["training_sequence_tokens"]<=16000
    cfg={"model":str(base),"train_data":str(root/"sft_train.jsonl"),"validation_data":str(root/"sft_validation.jsonl"),
      "output_dir":str(root/"sft"),"max_length":2048,"lora_rank":4,"lora_alpha":8,"learning_rate":.001,
      "batch_size":1,"gradient_accumulation":2,"allow_cpu":True,"gradient_checkpointing":True,
      "max_steps":2,"eval_steps":1,"save_steps":1,"logging_steps":1}
    initial=train(cfg);cfg["max_steps"]=3;resumed=train(cfg,resume="latest")
    weights=load_file(root/"sft/adapter_model.safetensors")
    assert any(v.abs().sum()>0 for k,v in weights.items() if "lora_B" in k)
    dpo={**cfg,"init_adapter":str(root/"sft"),"output_dir":str(root/"dpo"),"max_steps":2,"learning_rate":.0001,"beta":.1,
         "train_data":str(root/"ref_train.jsonl"),"validation_data":str(root/"ref_validation.jsonl")}
    for split in ("train","validation"):cache_reference(dpo,root/f"prefs_{split}.jsonl",root/f"ref_{split}.jsonl")
    result=train(dpo,stage="dpo")
    assert abs(result["eval_loss"]-.693147)>1e-6
    write_jsonl(root/"test_two.jsonl",list(read_jsonl(root/"raw/test.jsonl"))[:2])
    collection=collect([root/"test_two.jsonl"],root/"tiny_local.jsonl",model=str(base),adapter=str(root/"dpo"),budgets=[8],allow_cpu=True)
    export_model(str(base),str(root/"dpo"),str(root/"merged"))
    report={"passed":True,"kind":"tiny random Qwen3 CPU execution with contrast curriculum",
            "sft":initial,"resumed_sft":resumed,"dpo":result,"local_collection":collection,"merged_export":True,
            "not_tested":["8B CUDA training","4-bit CUDA training","MareNostrum submission","live Nebius API","VLM GPU training","capability gains"]}
    write_json(root/"smoke_report.json",report);print(report)


if __name__=="__main__":main()
