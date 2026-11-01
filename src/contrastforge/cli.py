import argparse
import json
from pathlib import Path
from probeforge.io import canonical,read_jsonl,write_json,write_jsonl


def parser():
    p=argparse.ArgumentParser(description="ProbeForge 0.2: verified contrast training")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("generate","lic-generate"):
        s=sub.add_parser(name);s.add_argument("--out",required=True);s.add_argument("--groups",type=int,default=1000 if name=="generate" else 60)
        s.add_argument("--seed",type=int,default=42);s.add_argument("--languages",nargs="+",default=["en","es","ca","ar","lb"])
        if name=="generate": s.add_argument("--metamorphic",action="store_true",help="add context_academic/context_business/expansion views");s.add_argument("--status-diverse",action="store_true",help="add boundary_alt (second missing fact) and infeasible views")
    s=sub.add_parser("prepare");s.add_argument("--inputs",nargs="+",required=True);s.add_argument("--out",required=True)
    s.add_argument("--strategy",choices=["mined","translated","random","no_counterfactual"],default="mined")
    s.add_argument("--rollouts",nargs="*",default=[]);s.add_argument("--seed",type=int,default=42)
    s.add_argument("--token-budget",type=int);s.add_argument("--tokenizer",dest="tokenizer_path")
    s.add_argument("--allow-synthetic",action="store_true");s.add_argument("--background-fraction",type=float,default=.2)
    s.add_argument("--targets",choices=["gold","worked"],default="gold",help="worked = derivation-first completions (facts, explanation with formula trace, status, answer)")
    s=sub.add_parser("collect");s.add_argument("--inputs",nargs="+",required=True);s.add_argument("--out",required=True)
    s.add_argument("--backend",choices=["local","api","mock"],default="local");s.add_argument("--model")
    s.add_argument("--adapter");s.add_argument("--revision",default="main");s.add_argument("--base-url")
    s.add_argument("--key-env",default="NEBIUS_API_KEY");s.add_argument("--budgets",type=int,nargs="+",default=[512])
    s.add_argument("--samples",type=int,default=1);s.add_argument("--seed",type=int,default=42)
    s.add_argument("--temperature",type=float,default=0);s.add_argument("--quantize",action="store_true")
    s.add_argument("--allow-cpu",action="store_true");s.add_argument("--nvml-indices",type=int,nargs="+")
    s.add_argument("--vision",action="store_true");s.add_argument("--only-ids",help="JSON array produced by prepare")
    s.add_argument("--thinking",action="store_true",help="enable the chat template thinking mode (Qwen3); <think> block is stripped before verification")
    s.add_argument("--api-extra",help="JSON file with supported provider decoding options")
    s=sub.add_parser("evaluate");s.add_argument("--inputs",nargs="+",required=True);s.add_argument("--rollouts",nargs="+",required=True)
    s.add_argument("--out",required=True);s.add_argument("--allow-synthetic",action="store_true")
    s=sub.add_parser("compare");s.add_argument("--reports",nargs="+",required=True);s.add_argument("--out",required=True)
    s=sub.add_parser("import-legacy");s.add_argument("--paths",nargs="+",required=True);s.add_argument("--out",required=True)
    s=sub.add_parser("import-retention");s.add_argument("--manifest",required=True);s.add_argument("--out",required=True)
    s=sub.add_parser("preflight");s.add_argument("--inputs",nargs="+",required=True);s.add_argument("--tokenizer");s.add_argument("--max-length",type=int,default=2048);s.add_argument("--out")
    for name in ("train","cache-reference","train-vision"):
        s=sub.add_parser(name);s.add_argument("--config",required=True)
        for option in ("model","train-data","validation-data","output-dir","init-adapter"):
            s.add_argument("--"+option)
        if name=="cache-reference":s.add_argument("--input",required=True);s.add_argument("--output",required=True)
        else:
            s.add_argument("--resume",nargs="?",const="latest");s.add_argument("--max-steps",type=int)
            if name=="train":s.add_argument("--stage",choices=["sft","dpo"],default="sft")
    s=sub.add_parser("export");s.add_argument("--model",required=True);s.add_argument("--adapter",required=True);s.add_argument("--out",required=True);s.add_argument("--revision",default="main")
    s=sub.add_parser("download");s.add_argument("--model",default="Qwen/Qwen3-8B");s.add_argument("--out",required=True);s.add_argument("--revision",default="main")
    s=sub.add_parser("vision-prepare");s.add_argument("--manifest",required=True);s.add_argument("--image-root",required=True);s.add_argument("--out",required=True)
    s.add_argument("--languages",nargs="+",default=["en","es","ca","ar","lb"])
    s=sub.add_parser("lic-baselines");s.add_argument("--input",required=True);s.add_argument("--out",required=True)
    s=sub.add_parser("accept-teacher");s.add_argument("--inputs",nargs="+",required=True);s.add_argument("--rollouts",nargs="+",required=True);s.add_argument("--out",required=True)
    s=sub.add_parser("demo");s.add_argument("--out",required=True)
    sub.add_parser("doctor")
    return p


def main(argv=None):
    args=vars(parser().parse_args(argv));cmd=args.pop("command")
    if cmd=="generate":
        from .dataset import generate
        result=generate(**args)
    elif cmd=="lic-generate":
        from .lic import generate_lic
        result=generate_lic(**args)
    elif cmd=="prepare":
        from .curriculum import prepare
        result=prepare(**args)
    elif cmd=="collect":
        from .collect import collect
        if args["backend"]!="mock" and not args["model"]:raise ValueError("--model is required")
        if args["backend"]=="api" and not args["base_url"]:raise ValueError("--base-url is required; copy your current Nebius endpoint")
        if args["only_ids"]:args["only_ids"]=json.loads(Path(args["only_ids"]).read_text())
        if args["api_extra"]:args["api_extra"]=json.loads(Path(args["api_extra"]).read_text())
        result=collect(**args)
    elif cmd=="evaluate":
        from .evaluate import evaluate
        result=evaluate(**args)
    elif cmd=="compare":
        from .evaluate import compare
        result=compare(args["reports"],args["out"])
    elif cmd=="import-legacy":
        from .dataset import import_legacy
        result=import_legacy(**args)
    elif cmd=="import-retention":
        from .dataset import import_retention
        result=import_retention(args["manifest"],args["out"])
    elif cmd=="preflight":
        from .dataset import audit_rows,gold,sft_row
        from .verify import verify
        rows=[r for p in args["inputs"] for r in read_jsonl(p)];result=audit_rows(rows)
        for row in rows:
            checked=verify(row,canonical(gold(row)))
            if not checked["correct"]:raise ValueError(f"Oracle target failed verification: {row['id']}")
        if args["tokenizer"]:
            from transformers import AutoTokenizer
            from probeforge.training_data import encode_row
            tok=AutoTokenizer.from_pretrained(args["tokenizer"],trust_remote_code=False)
            lengths=[len(encode_row(tok,sft_row(r),args["max_length"])["input_ids"]) for r in rows if r["kind"]!="vlm"]
            result.update(max_tokens=max(lengths,default=0),mean_tokens=sum(lengths)/max(1,len(lengths)))
        if args["out"]:write_json(args["out"],result)
    elif cmd in ("train","cache-reference","train-vision"):
        from probeforge.train import load_config,train,cache_reference
        cfg=load_config(args.pop("config"),{k:args.get(k) for k in ("model","train_data","validation_data","output_dir","init_adapter","max_steps")})
        if cmd=="cache-reference":result=cache_reference(cfg,args["input"],args["output"])
        elif cmd=="train-vision":
            from .vision_train import train_vision
            result=train_vision(cfg,args["resume"])
        else:result=train(cfg,args["stage"],args["resume"])
    elif cmd=="export":
        from probeforge.train import export_model
        export_model(args["model"],args["adapter"],args["out"],args["revision"]);result={"exported":args["out"]}
    elif cmd=="download":
        from huggingface_hub import HfApi,snapshot_download
        sha=HfApi().model_info(args["model"],revision=args["revision"]).sha
        path=snapshot_download(args["model"],revision=sha,local_dir=args["out"],
                allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.jinja","*.tiktoken","LICENSE*"])
        write_json(Path(path)/"download_provenance.json",{"model":args["model"],"revision":sha})
        result={"model":args["model"],"revision":sha,"path":path}
    elif cmd=="vision-prepare":
        from .vision import prepare_vision
        result=prepare_vision(args["manifest"],args["image_root"],args["out"],args["languages"])
    elif cmd=="lic-baselines":
        from .lic import baseline_report
        records=baseline_report(list(read_jsonl(args["input"])));write_jsonl(args["out"],records);result={"records":len(records)}
    elif cmd=="accept-teacher":
        from .curriculum import checked_records
        from .dataset import sft_row
        rows=[r for p in args["inputs"] for r in read_jsonl(p)]
        if any(r["split"] not in ("train","validation") for r in rows):raise ValueError("Teacher data cannot include evaluation splits")
        records,_=checked_records(rows,args["rollouts"])
        accepted=[]
        for row in rows:
            good=[r for r in records.get(row["id"],[]) if r["verification"]["correct"]]
            if good:
                candidate=min(good,key=lambda r:len(r["response"]))
                accepted.append({**sft_row(row,candidate["response"]),"teacher_run":candidate["run_fingerprint"],"language_reviewed":False})
        write_jsonl(args["out"],accepted);result={"accepted":len(accepted),"note":"Semantic checks passed; natural-language explanations require review"}
    elif cmd=="demo":
        from .dataset import generate
        from .collect import collect
        from .curriculum import prepare
        from .evaluate import evaluate
        root=Path(args["out"])
        generate(root/"dataset",groups=6,languages=["en","es","ar"])
        inputs=[root/"dataset/train.jsonl",root/"dataset/validation.jsonl"]
        collect(inputs,root/"student_fixture.jsonl",backend="mock")
        prepare(inputs,root/"curriculum",rollouts=[root/"student_fixture.jsonl"],allow_synthetic=True)
        collect([root/"dataset/test.jsonl"],root/"test_fixture.jsonl",backend="mock")
        result=evaluate([root/"dataset/test.jsonl"],[root/"test_fixture.jsonl"],root/"report.json",allow_synthetic=True)
        result={"completed":True,"synthetic":True,"report":str(root/"report.json"),"not_model_results":True}
    else:
        import importlib.util,platform
        result={"python":platform.python_version(),"platform":platform.platform(),
                "optional_packages":{p:bool(importlib.util.find_spec(p)) for p in ("torch","transformers","peft","pynvml","PIL")}}
        if result["optional_packages"]["torch"]:
            import torch
            result.update(cuda=torch.cuda.is_available(),gpu_count=torch.cuda.device_count())
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
