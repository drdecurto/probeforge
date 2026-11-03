"""Merge a PEFT LoRA adapter into Kolibri-1 safetensors shards, keeping the original key names/layout.
Untouched shards and side files are symlinked. Usage: merge_lora.py ADAPTER_DIR OUT_DIR"""
import json,os,sys,math,torch
from safetensors import safe_open
from safetensors.torch import save_file
M="/gpfs/scratch/bsc21/bsc021885/cf/models/Kolibri-1-BF16"; A,O=sys.argv[1],sys.argv[2]; os.makedirs(O,exist_ok=True)
cfg=json.load(open(A+"/adapter_config.json")); r=cfg["r"]; sc=cfg["lora_alpha"]/(math.sqrt(r) if cfg.get("use_rslora") else r)
assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern")
lo={}
with safe_open(A+"/adapter_model.safetensors","pt") as f:
    for k in f.keys():
        base,ab=k.split(".lora_"); tgt=base.replace("base_model.model.","",1)+".weight"; lo.setdefault(tgt,{})[ab[0]]=f.get_tensor(k)
idx=json.load(open(M+"/model.safetensors.index.json"))["weight_map"]
miss=[t for t in lo if t not in idx]; assert not miss, miss[:5]
byshard={}
for t in lo: byshard.setdefault(idx[t],[]).append(t)
for fn in os.listdir(M):
    if fn.endswith(".safetensors") and fn in byshard: continue
    d=os.path.join(O,fn)
    if not os.path.lexists(d): os.symlink(os.path.join(M,fn),d)
n=0; maxrel=0
for sh,ts in sorted(byshard.items()):
    out={}
    with safe_open(os.path.join(M,sh),"pt") as f:
        meta=f.metadata()
        for k in f.keys(): out[k]=f.get_tensor(k)
    for t in ts:
        W=out[t]; d=(lo[t]["B"].float()@lo[t]["A"].float())*sc; assert d.shape==W.shape,(t,d.shape,W.shape)
        maxrel=max(maxrel,(d.norm()/W.float().norm()).item()); out[t]=(W.float()+d).to(W.dtype); n+=1
    tmp=os.path.join(O,sh+".tmp"); save_file(out,tmp,metadata=meta or {"format":"pt"}); os.replace(tmp,os.path.join(O,sh))
print(f"merged {n} tensors in {len(byshard)} shards, scale {sc}, max |dW|/|W| {maxrel:.4f}")
