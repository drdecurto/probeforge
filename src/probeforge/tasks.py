"""Finite hypothesis environments. No generated source code is executed.

Each group contains all counterfactual worlds, sharing the complete public prompt.
Candidate rules are intentionally public: v0.1 measures experiment selection and
evidence use in a finite, known hypothesis space, not open-ended program synthesis.
"""
import itertools
import random
from collections import Counter
from pathlib import Path

from .io import digest, write_json, write_jsonl, file_hash

TRAIN_FAMILIES = ("affine", "quadratic", "sequence_sum", "sequence_weighted", "battery_linear", "battery_efficiency")
OOD_FAMILIES = ("absolute", "threshold", "modular", "sequence_clipped", "battery_standby", "battery_saturation")


def evaluate_rule(rule, x):
    f, p = rule["family"], rule["parameters"]
    a, b, c = p
    if f == "affine":
        return a*x + b*max(0, x) + c*min(0, x)
    if f == "quadratic":
        return a*x + b*x*x + c*max(0, x-2)
    if f == "absolute":
        return a*x + b*abs(x) + c*max(0, abs(x)-2)
    if f == "threshold":
        return a*x + b*int(x > 2) + c*int(x < -2)
    if f == "modular":
        return a*x + b*(x % 2) + c*(x % 3)
    if f == "sequence_sum":
        return a*sum(x) + b*sum(v > 0 for v in x) + c*sum(v < 0 for v in x)
    if f == "sequence_weighted":
        return a*sum(x) + b*sum((i+1)*v for i, v in enumerate(x)) + c*len(set(x))
    if f == "sequence_clipped":
        return a*sum(max(-2, min(2, v)) for v in x) + b*len(x) + c*sum(v == 0 for v in x)
    # Normalized integer toy battery model: input = [state, charge, demand].
    s, charge, demand = x
    if f == "battery_linear":
        return s + a*charge - b*demand - c*int(charge > 0 and demand > 0)
    if f == "battery_efficiency":
        return s + (a*charge)//2 - b*demand - c*int(charge > 2)
    if f == "battery_standby":
        return s + a*charge - b*demand - c*int(charge == 0 and demand == 0)
    if f == "battery_saturation":
        return max(0, min(30, s + a*charge - b*demand - c*int(s > 20)))
    raise ValueError(f"Unknown family: {f}")


FORMULAS = {
    "affine": "a*x + b*max(0,x) + c*min(0,x)",
    "quadratic": "a*x + b*x*x + c*max(0,x-2)",
    "absolute": "a*x + b*abs(x) + c*max(0,abs(x)-2)",
    "threshold": "a*x + b*int(x>2) + c*int(x < -2)",
    "modular": "a*x + b*(x%2) + c*(x%3)",
    "sequence_sum": "a*sum(x) + b*count(v>0) + c*count(v<0)",
    "sequence_weighted": "a*sum(x) + b*sum((i+1)*x[i]) + c*count_distinct(x)",
    "sequence_clipped": "a*sum(max(-2,min(2,v))) + b*len(x) + c*count(v==0)",
    "battery_linear": "s + a*charge - b*demand - c*int(charge>0 and demand>0)",
    "battery_efficiency": "s + (a*charge)//2 - b*demand - c*int(charge>2)",
    "battery_standby": "s + a*charge - b*demand - c*int(charge==0 and demand==0)",
    "battery_saturation": "max(0,min(30,s + a*charge - b*demand - c*int(s>20)))",
}


def make_group(family, rng, budget):
    alternatives = [rng.sample(range(1, 8), 2) for _ in range(3)]
    candidates = [{"family": family, "parameters": list(p)} for p in itertools.product(*alternatives)]
    rng.shuffle(candidates)  # IDs carry no information about the hidden world.
    if family.startswith("sequence"):
        anchor = []
        probes = [[1], [-1], [0], [1, 1], [1, -1], [3, -2, 0], [2, 1, -3], [0, 0], [-3, -3]]
        targets = [rng.choices(range(-5, 6), k=rng.randint(2, 5)) for _ in range(3)]
    elif family.startswith("battery"):
        anchor = [10, 0, 0]
        if family == "battery_standby":
            anchor = [10, 1, 0]
            # A common initial observation is not required here: filter anchor below.
        probes = [[10, 0, 0], [10, 1, 0], [10, 0, 1], [10, 2, 1], [10, 3, 0], [25, 1, 1], [2, 0, 3], [20, 3, 2], [1, 1, 1]]
        targets = [[rng.randint(0, 30), rng.randint(0, 5), rng.randint(0, 5)] for _ in range(3)]
    else:
        anchor = 0
        probes = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        targets = rng.sample([-7, -6, -5, 5, 6, 7], 3)
    # Retain only a common observation; standby gets no initial observation.
    values = {evaluate_rule(rule, anchor) for rule in candidates}
    initial = [{"input": anchor, "output": next(iter(values))}] if len(values) == 1 else []
    rng.shuffle(probes)
    menu = [{"id": f"p{i}", "input": x, "cost": rng.choice([1, 1, 1, 2])} for i, x in enumerate(probes)]
    structure = {"candidates": sorted([r["parameters"] for r in candidates]), "family": family}
    group_id = digest(structure)[:20]
    # Same parameters always belong to one group, regardless of targets/seed.
    common = {"schema_version": 1, "group_id": group_id, "family": family,
              "candidates": candidates, "formula": FORMULAS[family], "initial": initial,
              "probes": menu, "targets": targets, "budget": budget}
    return [{**common, "id": f"{group_id}-w{i}", "truth_index": i} for i in range(len(candidates))]


def generate_dataset(out, groups=300, seed=42):
    if groups < 12:
        raise ValueError("Use at least 12 groups")
    root = Path(out)
    if (root / "manifest.json").exists():
        raise FileExistsError("Dataset exists; choose a new output directory")
    rng, seen = random.Random(seed), set()
    splits = {k: [] for k in ("train", "validation", "test_id", "test_ood")}
    # Exactly separated group counts; no reliance on random split occupancy.
    counts = {"train": max(6, int(groups*.65)), "validation": max(1, int(groups*.10)),
              "test_id": max(1, int(groups*.10))}
    counts["test_ood"] = groups - sum(counts.values())
    for split, count in counts.items():
        families = OOD_FAMILIES if split == "test_ood" else TRAIN_FAMILIES
        for j in range(count):
            while True:
                group = make_group(families[j % len(families)], rng, rng.choice([1, 2, 3, 4]))
                if group[0]["group_id"] not in seen:
                    seen.add(group[0]["group_id"])
                    break
            splits[split].extend({**task, "split": split} for task in group)
    for split, rows in splits.items():
        write_jsonl(root / f"{split}.jsonl", rows)
    manifest = {"seed": seed, "groups": groups, "version": 1,
                "note": "Private evaluator files contain truth_index. Policy receives public_state only.",
                "splits": {k: {"worlds": len(v), "families": dict(Counter(t['family'] for t in v)),
                               "sha256": file_hash(root / f"{k}.jsonl")} for k, v in splits.items()}}
    write_json(root / "manifest.json", manifest)
    return manifest
