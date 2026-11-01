from collections import defaultdict
from pathlib import Path
import random
from probeforge.io import canonical, digest, read_jsonl, write_json, write_jsonl
from .dataset import sft_row, gold, audit_rows
from .verify import verify


def checked_records(rows, paths, allow_synthetic=False):
    by_id={r["id"]:r for r in rows};records=defaultdict(list);seen=set();runs=set()
    for path in paths:
        for rec in read_jsonl(path):
            if rec["request_id"] in seen:raise ValueError("Duplicate rollout request")
            seen.add(rec["request_id"])
            row=by_id.get(rec["id"])
            if row is None:raise ValueError("Rollout contains an example outside this dataset")
            if rec["example_hash"]!=digest(row):raise ValueError("Rollout example changed")
            if rec.get("synthetic") and not allow_synthetic:
                raise ValueError("Synthetic fixtures require explicit --allow-synthetic and are not model evidence")
            checked=verify(row,rec["response"])
            checked["correct"] = checked["correct"] and rec.get("finish_reason")=="stop"
            records[row["id"]].append({**rec,"verification":checked})
            runs.add(rec["run_fingerprint"])
    return records,runs


def token_cost(tokenizer,row):
    from probeforge.training_data import encode_row
    return len(encode_row(tokenizer,row,10**7)["input_ids"])


def prepare(inputs,out,strategy="mined",rollouts=(),seed=42,token_budget=None,tokenizer_path=None,
            allow_synthetic=False,background_fraction=.2):
    rows=[r for p in inputs for r in read_jsonl(p)]
    audit_rows(rows)
    if any(r["kind"]=="vlm" for r in rows):
        raise ValueError("Vision rows use train-vision directly so images are retained")
    if any(r["split"] not in ("train","validation") for r in rows):
        raise ValueError("Training preparation refuses test and diagnostic inputs")
    if not 0<=background_fraction<=1:raise ValueError("Invalid background fraction")
    root=Path(out)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Curriculum output exists")
    records,runs=checked_records(rows,rollouts,allow_synthetic)
    if strategy=="mined" and not records:raise ValueError("Mining needs student rollouts")
    if strategy not in ("mined","translated","random","no_counterfactual"):
        raise ValueError("Unknown curriculum")
    if strategy=="mined" and len(runs)!=1:
        raise ValueError("Mine one student run at a time; do not mix checkpoints or decoding plans")
    groups=defaultdict(list)
    for r in rows:
        if r["split"]=="train":groups[r["group_id"]].append(r)
    priority={};diagnostics=[]
    for group,members in groups.items():
        outcomes=[];missing=0
        for row in members:
            rr=records.get(row["id"],[])
            if not rr:missing+=1;continue
            # Pass@1-style first sample at largest collected budget; shorter budgets do not manufacture a reasoning gap.
            rec=min(rr,key=lambda r:(-r["budget"],r["sample"]))
            if rec["finish_reason"] in ("error",None):missing+=1;continue
            outcomes.append((row,rec["verification"]["correct"]))
        solved=sum(ok for _,ok in outcomes)
        failed=len(outcomes)-solved
        boundary_fail=sum(not ok and r["view"] in ("boundary","counterfactual","rewire") for r,ok in outcomes)
        partial=bool(solved and failed)
        # Complete coverage is required to call a group a measured consistency failure.
        score=(2*int(partial)+boundary_fail/max(1,len(outcomes))) if missing==0 else 0
        priority[group]=score
        diagnostics.append({"group_id":group,"score":score,"correct_views":solved,
                            "failed_views":failed,"missing_views":missing,"eligible_for_mining":missing==0})
    train=[r for r in rows if r["split"]=="train"]
    val=[r for r in rows if r["split"]=="validation"]
    if not train or not val:raise ValueError("Provide both train and validation inputs")
    if strategy=="translated":train=[r for r in train if r["view"]=="base"]
    if strategy=="no_counterfactual":train=[r for r in train if r["view"] not in ("counterfactual","boundary","rewire")]
    rng=random.Random(seed);rng.shuffle(train)
    if strategy=="mined":
        hard=[r for r in train if priority[r["group_id"]]>0]
        if not hard:raise ValueError("No fully observed repairable groups: collect missing views, increase difficulty, or use random curriculum")
        # Sampling weights preserve the group as the sampling unit; later expand all its views.
        group_order=list(groups);rng.shuffle(group_order)
        hard_groups=sorted((g for g in group_order if priority[g]>0),key=lambda g:-priority[g])
        background=[g for g in group_order if priority[g]==0]
        take=round(len(hard_groups)*background_fraction/max(1e-9,1-background_fraction))
        selected=hard_groups+background[:take]
        train=[r for g in selected for r in groups[g]]
        rng.shuffle(train)
    supervised=[sft_row(r) for r in train]
    tokenizer=None
    if token_budget is not None:
        if token_budget<1 or not tokenizer_path:raise ValueError("Token budget requires a tokenizer and positive budget")
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=False)
        units=defaultdict(list)
        for row in supervised:units[row["group_id"]].append(row)
        units=list(units.values())
        costs=[sum(token_cost(tokenizer,r) for r in unit) for unit in units]
        if min(costs)>token_budget:raise ValueError("Token budget smaller than every complete problem group")
        selected=[];total=0;cycles=0
        # Whole examples only. Repetition is intentional, recorded, and identical in policy across arms.
        while True:
            added=0
            for unit,cost in zip(units,costs):
                if total+cost<=token_budget:
                    offset=len(selected)
                    selected.extend({**row,"exposure_index":offset+i} for i,row in enumerate(unit))
                    total+=cost;added+=1
            cycles+=1
            if added==0 or token_budget-total<min(costs):break
        supervised=selected
    else:total=None;cycles=1
    preferences={"train":[],"validation":[]};failed_ids=[]
    selected_ids={r["id"] for r in supervised}
    for row in rows:
        rr=records.get(row["id"],[])
        if row["split"]=="train" and row["id"] not in selected_ids:continue
        bad=[r for r in rr if not r["verification"]["correct"] and r.get("finish_reason")=="stop" and r["response"].strip()]
        if bad:
            reject=sorted(bad,key=lambda r:(-r["budget"],r["sample"]))[0]
            pref={k:v for k,v in sft_row(row).items() if k!="completion"}
            pref.update(chosen=canonical(gold(row)),rejected=reject["response"],preference_reason="verified_correctness")
            preferences[row["split"]].append(pref)
            if row["split"]=="train":failed_ids.append(row["id"])
        # Optional brevity preferences only compare independently correct outputs under the same prompt.
        good=[r for r in rr if r["verification"]["correct"] and isinstance(r.get("usage",{}).get("completion_tokens"),int)]
        if len(good)>1:
            short=min(good,key=lambda r:r["usage"]["completion_tokens"])
            long=max(good,key=lambda r:r["usage"]["completion_tokens"])
            if long["usage"]["completion_tokens"]>1.5*short["usage"]["completion_tokens"] and short["response"]!=long["response"]:
                pref={k:v for k,v in sft_row(row).items() if k!="completion"}
                pref.update(chosen=short["response"],rejected=long["response"],preference_reason="verified_brevity_unreviewed_language")
                # Separate file: requires language/explanation review before mixing into DPO.
                preferences.setdefault("brevity_review",[]).append(pref)
    write_jsonl(root/"sft_train.jsonl",supervised)
    write_jsonl(root/"sft_validation.jsonl",[sft_row(r) for r in val])
    for split,prefs in preferences.items():write_jsonl(root/f"preferences_{split}.jsonl",prefs)
    write_jsonl(root/"mining_diagnostics.jsonl",diagnostics)
    write_json(root/"repair_ids.json",sorted(set(failed_ids)))
    manifest={"strategy":strategy,"seed":seed,"synthetic_rollouts_allowed":allow_synthetic,
              "source_hash":digest(rows),"rollout_runs":sorted(runs),"training_rows":len(supervised),
              "unique_training_examples":len(selected_ids),"validation_rows":len(val),"exposure_cycles":cycles,
              "training_sequence_tokens":total,"requested_token_budget":token_budget,
              "token_budget_note":"Full prompt+completion tokens; no padding or GPU-FLOP equivalence implied. Train exactly one epoch.",
              "preference_counts":{k:len(v) for k,v in preferences.items()}}
    write_json(root/"manifest.json",manifest)
    return manifest
