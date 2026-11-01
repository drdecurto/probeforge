import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
from probeforge.io import canonical, digest, file_hash, read_jsonl, write_json
from .dataset import gold
from .verify import verify
from .energy import EnergyMeter, checked_nvml_indices


class APIBackend:
    def __init__(self, model, base_url, key_env="NEBIUS_API_KEY", timeout=int(os.environ.get("CF_API_TIMEOUT","120")), retries=2, api_extra=None):
        from urllib.parse import urlparse
        host=urlparse(base_url).hostname
        if not base_url.startswith("https://") and host not in ("localhost","127.0.0.1","::1"):
            raise ValueError("Remote API endpoints require HTTPS")
        self.model,self.url,self.key_env,self.timeout,self.retries=model,base_url.rstrip("/")+"/chat/completions",key_env,timeout,retries
        self.extra=api_extra or {}
        if not isinstance(self.extra,dict) or not set(self.extra)<={"chat_template_kwargs","reasoning_effort","top_p"}:
            raise ValueError("API extras allow only chat_template_kwargs, reasoning_effort, and top_p")
        if not os.environ.get(key_env) and host not in ("localhost","127.0.0.1","::1"):
            raise ValueError(f"Set {key_env}; credentials are never written to logs")

    def generate(self, row, budget, seed, temperature):
        prompt = row["prompt"]
        if row["kind"] == "vlm":
            from .vision import api_messages
            prompt=api_messages(row)
        payload={"model":self.model,"messages":prompt,"max_tokens":budget,"temperature":temperature,"seed":seed}
        payload.update(self.extra)
        attempts=[]
        for attempt in range(self.retries+1):
            start=time.perf_counter()
            req=urllib.request.Request(self.url, data=canonical(payload).encode(),
                   headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ.get(self.key_env,"")})
            try:
                with urllib.request.urlopen(req,timeout=self.timeout) as response:
                    result=json.load(response)
                choice=result["choices"][0]; text=choice["message"].get("content") or ""
                usage=result.get("usage",{})
                attempts.append({"status":"ok","seconds":time.perf_counter()-start})
                return {"response":text,"finish_reason":choice.get("finish_reason"),"usage":usage,
                        "attempts":attempts,"energy":{"joules":None,"method":"remote_api_not_measured"},
                        "server_model":result.get("model"),"system_fingerprint":result.get("system_fingerprint")}
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ValueError,KeyError) as error:
                status=getattr(error,"code",None)
                attempts.append({"status":"error","http_status":status,"error_type":type(error).__name__,
                                 "seconds":time.perf_counter()-start,"usage_unknown":True})
                retry = status in (429,500,502,503,504) or isinstance(error,(urllib.error.URLError,TimeoutError))
                if not retry or attempt==self.retries: break
                time.sleep(min(2**attempt,8))
        return {"response":"","finish_reason":"error","usage":{},"attempts":attempts,
                "energy":{"joules":None,"method":"remote_api_not_measured"}}


class LocalBackend:
    def __init__(self, model, adapter=None, revision="main", quantize=False, allow_cpu=False,
                 nvml_indices=None, vision=False, thinking=False):
        import torch
        self.vision=vision;self.thinking=thinking
        cfg={"model":model,"init_adapter":adapter,"revision":revision,"quantize":quantize,"allow_cpu":allow_cpu}
        if vision:
            from .vision_train import load_vision
            self.model,self.tokenizer,self.device=load_vision(cfg,trainable=False)
        else:
            from probeforge.train import load_model
            self.model,self.tokenizer,self.device=load_model(cfg,trainable=False)
        self.model.eval();self.model.config.use_cache=True
        self.indices=checked_nvml_indices(nvml_indices,self.device)
        self.warmed=False

    def generate(self,row,budget,seed,temperature):
        import torch
        from transformers import set_seed
        from probeforge.training_data import render_prompt
        set_seed(seed)
        if self.vision:
            from .vision_train import vision_inputs
            inputs=vision_inputs(self.tokenizer,row)
        else:
            inputs=self.tokenizer(render_prompt(self.tokenizer,row["prompt"],thinking=self.thinking),return_tensors="pt",add_special_tokens=False,return_token_type_ids=False)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        context=getattr(self.model.config,"max_position_embeddings",None)
        if context and inputs["input_ids"].shape[1]+budget>context:
            raise ValueError("Prompt plus output budget exceeds model context; no silent truncation")
        def sync():
            if self.device.startswith("cuda"):torch.cuda.synchronize()
        if not self.warmed:
            with torch.inference_mode(): self.model.generate(**inputs,max_new_tokens=2,do_sample=False)
            sync();self.warmed=True;set_seed(seed)
        generation={"max_new_tokens":budget,"do_sample":temperature>0}
        if temperature>0:generation.update(temperature=temperature,top_p=.95)
        sync();start=time.perf_counter()
        with EnergyMeter(self.indices) as meter:
            with torch.inference_mode():output=self.model.generate(**inputs,**generation)
            sync()
        elapsed=time.perf_counter()-start
        tokens=output[0,inputs["input_ids"].shape[1]:]
        tokenizer=self.tokenizer.tokenizer if self.vision else self.tokenizer
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        thinking_tokens=None;thinking_text=None
        if self.thinking:
            # Qwen3-style reasoning: strip the <think>...</think> block; only the final answer is verified.
            raw=text
            if "</think>" in raw:
                head,_,text=raw.partition("</think>")
                thinking_tokens=len(tokenizer.encode(head,add_special_tokens=False))
                thinking_text=head.replace("<think>","",1).strip()  # kept for reasoning-trace distillation
            else:
                text=""  # reasoning never closed within the budget -> treated as an incomplete answer
                thinking_tokens=len(tokens)
            text=text.strip()
        eos=self.model.generation_config.eos_token_id
        eos=[] if eos is None else ([eos] if isinstance(eos,int) else eos)
        finished=len(tokens)>0 and int(tokens[-1]) in eos
        return {"response":text,"finish_reason":"stop" if finished or len(tokens)<budget else "length",
                **({"thinking":thinking_text} if thinking_text is not None else {}),
                "usage":{"prompt_tokens":inputs["input_ids"].shape[1],"completion_tokens":len(tokens),"thinking_tokens":thinking_tokens},
                "attempts":[{"status":"ok","seconds":elapsed}],"energy":meter.result}


class MockBackend:
    """Deterministic software fixture. Never represents an LLM capability measurement."""
    def generate(self,row,budget,seed,temperature):
        out=gold(row)
        if row["kind"]=="math" and (row["language"] in ("ar","lb") or row["view"]=="counterfactual"):
            out["status"]="infeasible" if out["status"]=="solved" else "solved"
        return {"response":canonical(out),"finish_reason":"stop","usage":{},
                "attempts":[{"status":"synthetic_fixture","seconds":0}],
                "energy":{"joules":None,"method":"synthetic_fixture"}}


def collect(inputs,out,backend="local",model=None,adapter=None,revision="main",base_url=None,
            key_env="NEBIUS_API_KEY",budgets=(512,),samples=1,seed=42,temperature=0,
            quantize=False,allow_cpu=False,nvml_indices=None,vision=False,only_ids=None,api_extra=None,thinking=False):
    if samples<1 or any(x<1 for x in budgets) or len(set(budgets))!=len(budgets): raise ValueError("Invalid sampling plan")
    rows=[r for p in inputs for r in read_jsonl(p)]
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate input IDs")
    # The run fingerprint covers the FULL input set so that --only-ids shards of one run share it.
    full_input_hash=digest(rows)
    if only_ids is not None: rows=[r for r in rows if r["id"] in set(only_ids)]
    if not rows:raise ValueError("No examples to collect")
    if any(r["kind"]=="vlm" for r in rows) and backend=="local" and not vision:
        raise ValueError("VLM examples require --vision and a vision checkpoint")
    config={"backend":backend,"model":model,"revision":revision,"adapter":adapter,"base_url":base_url,
            "quantize":quantize,"vision":vision,"seed":seed,"temperature":temperature,
            "budgets":list(budgets),"samples":samples,"input_hash":full_input_hash,"synthetic":backend=="mock",
            "nvml_indices":nvml_indices,"api_extra":api_extra,"thinking":bool(thinking)}
    if backend=="local":
        from probeforge.train import model_fingerprint
        config["model_fingerprint"]=model_fingerprint(model,revision,adapter)
        if not Path(model).is_dir() and revision=="main":
            raise ValueError("Local collections require a downloaded snapshot or immutable revision")
    fingerprint=digest(config)
    out=Path(out);meta=Path(str(out)+".manifest.json")
    existing=[]
    if out.exists():
        if not meta.exists() or json.loads(meta.read_text())["fingerprint"]!=fingerprint:
            raise ValueError("Resume configuration or dataset differs; choose a fresh output")
        existing=list(read_jsonl(out))
    else:
        if meta.exists():raise ValueError("Manifest exists without output; choose a fresh output")
        write_json(meta,{"fingerprint":fingerprint,"config":config})
        out.parent.mkdir(parents=True,exist_ok=True);out.touch()
    done={r["request_id"] for r in existing}
    if len(done)!=len(existing):raise ValueError("Duplicate request records")
    if backend=="api": runner=APIBackend(model,base_url,key_env,api_extra=api_extra)
    elif backend=="local": runner=LocalBackend(model,adapter,revision,quantize,allow_cpu,nvml_indices,vision,thinking)
    elif backend=="mock": runner=MockBackend()
    else:raise ValueError("Unknown backend")
    written=0
    with open(out,"a",encoding="utf-8") as stream:
        for row in rows:
            for budget in budgets:
                for sample in range(samples):
                    rid=digest([fingerprint,row["id"],budget,sample])
                    if rid in done:continue
                    sample_seed=int(digest([seed,row["id"],budget,sample])[:8],16)
                    start=time.perf_counter()
                    result=runner.generate(row,budget,sample_seed,temperature)
                    request_seconds=time.perf_counter()-start
                    checked=verify(row,result["response"])
                    complete=result["finish_reason"]=="stop"
                    checked["semantic_correct_before_completion_gate"]=checked["correct"]
                    checked["correct"]=checked["correct"] and complete
                    record={"request_id":rid,"id":row["id"],"group_id":row["group_id"],
                            "example_hash":digest(row),"run_fingerprint":fingerprint,"split":row["split"],
                            "budget":budget,"sample":sample,"seed":sample_seed,"synthetic":backend=="mock",
                            "generation_request_seconds":request_seconds,
                            "elapsed_seconds":time.perf_counter()-start,**result,"verification":checked}
                    stream.write(canonical(record)+"\n");stream.flush();os.fsync(stream.fileno())
                    written+=1
                    if written%25==0:print(f"Collected {written} new requests",flush=True)
    return {"new_requests":written,"previous_requests":len(done),"synthetic":backend=="mock","output":str(out)}
