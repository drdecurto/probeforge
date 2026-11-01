"""Exact finite-hypothesis planning baseline, using public observations only."""
from collections import Counter, defaultdict
from functools import lru_cache
from .environment import consistent, predictions, rules, available_probes
from .tasks import evaluate_rule


def action_values(state, cost_penalty=0.02):
    rs, possible = rules(state), consistent(state)
    if not possible:
        raise ValueError("Observations contradict every candidate")
    signatures = [tuple(predictions(state, i)) for i in range(len(rs))]
    probes = available_probes(state)

    def stop(indices):
        counts = Counter(signatures[i] for i in indices)
        signature, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        return count / len(indices), list(signature)

    @lru_cache(None)
    def value(indices, budget, unused):
        best = stop(indices)[0]
        for j in unused:
            p = probes[j]
            if p["cost"] > budget:
                continue
            buckets = defaultdict(list)
            for i in indices:
                buckets[evaluate_rule(rs[i], p["input"])].append(i)
            if len(buckets) == 1:
                continue
            rest = tuple(k for k in unused if k != j)
            q = -cost_penalty*p["cost"] + sum(len(v)/len(indices)*value(tuple(v), budget-p["cost"], rest) for v in buckets.values())
            best = max(best, q)
        return best

    v, answer = stop(possible)
    result = [{"action": {"action": "answer", "predictions": answer}, "value": v}]
    for j, p in enumerate(probes):
        buckets = defaultdict(list)
        for i in possible:
            buckets[evaluate_rule(rs[i], p["input"])].append(i)
        rest = tuple(k for k in range(len(probes)) if k != j)
        q = -cost_penalty*p["cost"] + sum(len(v)/len(possible)*value(tuple(v), state["remaining_budget"]-p["cost"], rest) for v in buckets.values())
        result.append({"action": {"action": "probe", "probe_id": p["id"]}, "value": q})
    return sorted(result, key=lambda row: -row["value"])
