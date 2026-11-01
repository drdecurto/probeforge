from collections import defaultdict
from pathlib import Path
import random
from probeforge.io import read_jsonl,write_json,digest
from .curriculum import checked_records


def average(xs):return sum(xs)/len(xs) if xs else None


def interval(group_values,seed=42,replicates=1000):
    if len(group_values)<2:return None
    r=random.Random(seed);values=list(group_values)
    samples=sorted(average([r.choice(values) for _ in values]) for _ in range(replicates))
    return [samples[int(.025*replicates)],samples[min(replicates-1,int(.975*replicates))]]


def evaluate(inputs,rollouts,out,allow_synthetic=False):
    rows=[r for p in inputs for r in read_jsonl(p)]
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate evaluation IDs")
    records,runs=checked_records(rows,rollouts,allow_synthetic)
    if len(runs)!=1:raise ValueError("Evaluate one checkpoint/decoding configuration at a time")
    budgets=sorted({r["budget"] for rr in records.values() for r in rr})
    report={"synthetic":any(r.get("synthetic") for rr in records.values() for r in rr),
            "run_fingerprint":next(iter(runs)),"dataset_hash":digest(sorted(rows,key=lambda r:r["id"])),
            "expected_examples":len(rows),"budgets":{}}
    for budget in budgets:
        groups=defaultdict(list);langs=defaultdict(list);views=defaultdict(list);families=defaultdict(list)
        statuses=defaultdict(list);per_group={};used=[];missing=0;truncated=0;all_samples=[]
        for row in rows:
            rr=[r for r in records.get(row["id"],[]) if r["budget"]==budget]
            all_samples.extend(rr)
            first=next((r for r in rr if r["sample"]==0),None)
            if first is None:missing+=1
            ok=bool(first and first["verification"]["correct"])
            if first:
                used.append(first);truncated+=first["finish_reason"]=="length"
                status=first["verification"].get("expected_status")
                if status:statuses[status].append(ok)
            groups[row["group_id"]].append(ok);langs[row["language"]].append(ok)
            views[row["view"]].append(ok);families[row["family"]].append(ok)
        for group,vals in groups.items():per_group[group]=average(vals)
        group_all=[all(xs) for xs in groups.values()]
        energies=[r.get("energy",{}).get("joules") for r in used]
        energy_complete=missing==0 and bool(energies) and all(x is not None for x in energies)
        correct=sum(r["verification"]["correct"] for r in used)
        attempts=[a for r in used for a in r.get("attempts",[])]
        tokens=[r.get("usage",{}).get("completion_tokens") for r in used]
        regret=[r["verification"]["regret"] for r in used if "regret" in r["verification"]]
        metrics={"pass_at_1":correct/len(rows),"missing_requests":missing,"length_stops":truncated,
          "per_language":{k:average(v) for k,v in langs.items()},"worst_language":min(average(v) for v in langs.values()),
          "per_view":{k:average(v) for k,v in views.items()},"per_family":{k:average(v) for k,v in families.items()},
          "per_status_observed":{k:average(v) for k,v in statuses.items()},
          "all_views_correct":average(group_all),"group_mean_accuracy_ci95":interval(list(per_group.values())),
          "ci_scope":"root-world bootstrap conditional on the included families; not unseen-family uncertainty",
          "mean_completion_tokens":average([v for v in tokens if v is not None]),
          "token_usage_coverage":sum(v is not None for v in tokens)/len(rows),
          "mean_request_latency_seconds":average([r.get("generation_request_seconds",r["elapsed_seconds"]) for r in used]),
          "actual_api_or_generation_attempts":len(attempts),"failed_attempts":sum(a["status"]=="error" for a in attempts),
          "unknown_usage_attempts":sum(bool(a.get("usage_unknown")) for a in attempts),
          "total_joules":sum(energies) if energy_complete else None,
          "joules_per_correct_answer":sum(energies)/correct if energy_complete and correct else None,
          "energy_coverage":sum(x is not None for x in energies)/len(rows),
          "mean_policy_regret_valid_only":average(regret),"policy_regret_coverage":len(regret)/len(rows),
          "group_scores":per_group,"all_collected_samples":len(all_samples)}
        report["budgets"][str(budget)]=metrics
    write_json(out,report);return report


def compare(report_paths,out):
    import json
    reports=[json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    if len(reports)<2:raise ValueError("Need baseline and at least one treatment report")
    comparisons=[]
    for report,path in zip(reports[1:],report_paths[1:]):
        if report["dataset_hash"]!=reports[0]["dataset_hash"]:raise ValueError("Evaluation examples differ")
        if report["synthetic"]!=reports[0]["synthetic"]:raise ValueError("Do not compare synthetic and real reports")
        for budget,base in reports[0]["budgets"].items():
            if budget not in report["budgets"]:continue
            other=report["budgets"][budget]
            if set(base["group_scores"])!=set(other["group_scores"]):raise ValueError("Evaluation group sets differ")
            diffs=[other["group_scores"][g]-v for g,v in base["group_scores"].items()]
            comparisons.append({"treatment":str(path),"budget":budget,"mean_paired_group_difference":average(diffs),
                                "paired_ci95":interval(diffs),"groups":len(diffs)})
    result={"baseline":str(report_paths[0]),"comparisons":comparisons};write_json(out,result);return result
