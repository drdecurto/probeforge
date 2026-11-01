"""Prepare reviewable, independent training configurations; never submits jobs."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("--base",required=True);p.add_argument("--curricula-root",default="data/curricula")
p.add_argument("--out",default="configs/matrix");p.add_argument("--arms",nargs="+",default=["translated","random","mined","no_counterfactual"])
p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44]);a=p.parse_args()
root=Path(a.out)
if root.exists() and any(root.iterdir()):raise FileExistsError("Matrix directory exists")
root.mkdir(parents=True,exist_ok=True);base=json.loads(Path(a.base).read_text());records=[]
for arm in a.arms:
    manifest=json.loads((Path(a.curricula_root)/arm/"manifest.json").read_text())
    if manifest["training_sequence_tokens"] is None:raise ValueError("Prepare every arm with a tokenizer and token budget")
    for seed in a.seeds:
        cfg={**base,"seed":seed,"epochs":1,"max_steps":-1,
             "train_data":str(Path(a.curricula_root)/arm/"sft_train.jsonl"),
             "validation_data":str(Path(a.curricula_root)/arm/"sft_validation.jsonl"),
             "output_dir":f"runs/{arm}_seed{seed}"}
        idx=len(records);(root/f"run_{idx}.json").write_text(json.dumps(cfg,indent=2)+"\n")
        records.append({"index":idx,"arm":arm,"seed":seed,"sequence_tokens":manifest["training_sequence_tokens"]})
tokens=[r["sequence_tokens"] for r in records]
report={"runs":records,"token_spread_fraction":(max(tokens)-min(tokens))/max(tokens),
        "note":"Inspect token spread; grouping prevents exact equality. Keep one epoch and compare actual runtimes."}
(root/"manifest.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
