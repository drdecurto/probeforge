import csv
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .environment import run_episode
from .io import canonical, digest, read_jsonl, write_json, write_jsonl, file_hash


def mean(values):
    return statistics.fmean(values) if values else 0.0


def interval(values, seed=42, samples=1000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [boots[int(samples*.025)], boots[min(samples-1, int(samples*.975))]]


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["split"], row["budget"], "all")].append(row)
        buckets[(row["split"], row["budget"], row["family"])].append(row)
    report = []
    for (split, budget, family), group_rows in sorted(buckets.items()):
        groups = defaultdict(list)
        for r in group_rows:
            groups[r["group_id"]].append(r)
        accuracies = [mean([float(r["success"]) for r in g]) for g in groups.values()]
        tokens_known = all(r.get("token_usage_available", False) for r in group_rows)
        report.append({"split": split, "budget": budget, "family": family, "worlds": len(group_rows),
                       "groups": len(groups), "success_rate": mean(accuracies), "group_bootstrap_95ci": interval(accuracies),
                       "all_worlds_solved_rate": mean([float(all(r["success"] for r in g)) for g in groups.values()]),
                       "tool_calls": mean([r["tool_calls"] for r in group_rows]),
                       "budget_spent": mean([r["budget_spent"] for r in group_rows]),
                       "invalid_episode_rate": mean([float(r["invalid_actions"] > 0) for r in group_rows]),
                       "completion_tokens": mean([r["completion_tokens"] for r in group_rows]) if tokens_known else None,
                       "prompt_tokens": mean([r["prompt_tokens"] for r in group_rows]) if tokens_known else None,
                       "token_usage_complete": tokens_known,
                       "seconds_per_episode": mean([r["seconds"] for r in group_rows])})
    return report


def evaluate(tasks_path, out, policy, budgets=(0, 1, 2, 4), max_groups=0, seed=42, keep_traces=True):
    root = Path(out)
    if (root / "episodes.jsonl").exists():
        raise FileExistsError("Evaluation exists; use a new directory")
    root.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for task in read_jsonl(tasks_path):
        groups[task["group_id"]].append(task)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    ids = ids[:max_groups or None]
    rows = []
    with open(root / "episodes.jsonl", "w", encoding="utf-8") as stream:
        for gid in ids:
            for budget in budgets:
                for task in groups[gid]:
                    start = time.monotonic()
                    r = run_episode(task, policy, budget=budget, seed=seed+int(digest(gid)[:8], 16))
                    r.update(budget=budget, seconds=time.monotonic()-start)
                    if not keep_traces:
                        r.pop("decisions", None)
                    stream.write(canonical(r)+"\n")
                    stream.flush()
                    rows.append(r)
    report = summarize(rows)
    write_json(root / "summary.json", report)
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as stream:
        if report:
            writer = csv.DictWriter(stream, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    write_json(root / "manifest.json", {"source_sha256": file_hash(tasks_path), "seed": seed,
               "budgets": list(budgets), "groups": ids, "completed": True,
               "note": "CI resamples entire counterfactual groups, not individual worlds."})
    return [r for r in report if r["family"] == "all"]


def compare(a_path, b_path, out):
    a, b = list(read_jsonl(a_path)), list(read_jsonl(b_path))
    key = lambda r: (r["task_id"], r["budget"])
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    if aa.keys() != bb.keys() or len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("Comparisons require exactly matching task/budget sets, without duplicates")
    buckets = defaultdict(lambda: defaultdict(list))
    for k in aa:
        x, y = aa[k], bb[k]
        if x.get("target") != y.get("target") and "target" in x and "target" in y:
            raise ValueError("Evaluation targets differ")
        buckets[(x["split"], x["budget"])][x["group_id"]].append(float(y["success"])-float(x["success"]))
    result = []
    for (split, budget), groups in sorted(buckets.items()):
        values = [mean(v) for v in groups.values()]
        result.append({"split": split, "budget": budget, "groups": len(values), "B_minus_A": mean(values),
                       "paired_group_bootstrap_95ci": interval(values)})
    write_json(out, result)
    return result
