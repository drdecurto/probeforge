"""LIC-inspired numerical curriculum, not a reproduction of the published social model.

All agents are synthetic numerical nodes. Utilities and penalties are stipulated,
not calibrated human behavior. The full state and objective are public to all baselines.
"""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import random
from pathlib import Path
from probeforge.io import digest, canonical, write_jsonl, write_json
from .worlds import number, rational
from .languages import LANGUAGES, index


def utility(state, policy):
    if set(policy) != {"targets", "intensity"}: raise ValueError("Policy schema")
    targets = policy["targets"]
    if not isinstance(targets, list) or any(type(x) is not int for x in targets): raise ValueError("Target IDs")
    n = len(state["benefit"])
    if len(targets) != len(set(targets)) or any(x < 0 or x >= n for x in targets): raise ValueError("Invalid targets")
    intensity = number(policy["intensity"])
    if intensity not in (F(1, 2), F(1)): raise ValueError("Intensity must be 1/2 or 1")
    if len(targets) > state["max_targets"]: raise ValueError("Too many targets")
    cost = sum(number(state["cost"][j])*intensity for j in targets)
    if cost > number(state["budget"]): raise ValueError("Budget exceeded")
    reached = set(targets)
    for a, b in state["edges"]:
        if a in targets: reached.add(b)
        if b in targets: reached.add(a)
    gain = sum(number(state["benefit"][j])*intensity for j in reached)
    fatigue = sum(number(state["fatigue"][j])*intensity**2 for j in targets)
    return gain - cost - fatigue


def candidates(state):
    for k in range(state["max_targets"]+1):
        for t in combinations(range(len(state["benefit"])), k):
            for intensity in ("1/2", "1"):
                policy = {"targets": list(t), "intensity": intensity}
                try: yield policy, utility(state, policy)
                except ValueError: pass


def optimum(state):
    return min(candidates(state), key=lambda z: (-z[1], len(z[0]["targets"]), z[0]["targets"], z[0]["intensity"]))


def heuristic(state, mode="greedy", frozen=None):
    if mode == "none": return {"targets": [], "intensity": "1/2"}
    n = len(state["benefit"])
    degree = [0]*n
    for a, b in state["edges"]: degree[a] += 1; degree[b] += 1
    best = ({"targets": [], "intensity": "1/2"}, F(0))
    for intensity in ("1/2", "1"):
        chosen = []
        for _ in range(state["max_targets"]):
            options = []
            for j in range(n):
                if j in chosen: continue
                p = {"targets": sorted(chosen+[j]), "intensity": intensity}
                try: score = utility(state, p)
                except ValueError: continue
                priority = score if mode == "greedy" else (degree[j] if frozen is None else -frozen.index(j))
                options.append((priority, -j, p, score))
            if not options: break
            _, _, p, score = max(options, key=lambda x: x[:2])
            chosen = p["targets"]
            if score > best[1]: best = p, score
    return best[0]


def make_state(topology, seed):
    r = random.Random(seed); n = 8
    if topology == "star": edges = [(0, j) for j in range(1, n)]
    elif topology == "ring": edges = [(j, (j+1)%n) for j in range(n)]
    elif topology == "grid": edges = [(j,j+1) for j in range(n) if j%4 != 3]+[(j,j+4) for j in range(4)]
    else: edges = [(a,b) for a in range(n) for b in range(a+1,n) if (a//4 == b//4 and r.random()<.7) or r.random()<.1]
    permutation = r.sample(range(n), n)
    edges = sorted(sorted((permutation[a], permutation[b])) for a,b in edges)
    return {"edges": edges, "benefit": [r.randint(1, 8) for _ in range(n)],
            "cost": [r.randint(1, 4) for _ in range(n)], "fatigue": [r.randint(0, 12) for _ in range(n)],
            "budget": 4, "max_targets": 2}


def instruction(lang):
    return ["Choose a policy for this synthetic network. Return only JSON with targets (distinct zero-based node IDs) and intensity (\"1/2\" or \"1\").",
            "Elige una política para esta red sintética. Devuelve solo JSON con targets (identificadores distintos de nodos desde cero) e intensity (\"1/2\" o \"1\").",
            "Tria una política per a aquesta xarxa sintètica. Retorna només JSON amb targets (identificadors diferents de nodes des de zero) i intensity (\"1/2\" o \"1\").",
            "اختر سياسة لهذه الشبكة الاصطناعية. أعد JSON فقط مع targets (معرّفات عقد مختلفة تبدأ من الصفر) وintensity (\"1/2\" أو \"1\").",
            "Wiel eng Politik fir dëst syntheetescht Netzwierk. Gëff nëmmen JSON mat targets (verschidden Node-IDen ab null) an intensity (\"1/2\" oder \"1\") zeréck."][index(lang)]


OBJECTIVE = ("The graph is undirected. reached = targets union their immediate neighbors (count each node once). "
             "utility = intensity*sum(benefit[j] for j in reached) - intensity*sum(cost[j] for j in targets) "
             "- intensity^2*sum(fatigue[j] for j in targets). Maximize utility. "
             "Constraints: intensity*sum(cost of targets)<=budget; number of targets<=max_targets. Empty targets are allowed. "
             "All fields describe the CURRENT state; recompute targets using this state.")


def generate_lic(out, groups=60, seed=42, languages=None):
    root = Path(out)
    if root.exists() and any(root.iterdir()): raise FileExistsError("Dataset directory is nonempty")
    languages = languages or list(LANGUAGES)
    if groups < 2: raise ValueError("Need at least two training groups")
    manifest = {"kind": "LIC-inspired synthetic coverage curriculum", "seed": seed,
                "not_a_reproduction": True, "objective_language": "English formal specification in every language",
                "splits": {}}
    for split, topologies in {"train": ("star", "ring"), "validation": ("grid",), "test": ("communities",)}.items():
        rows = []
        n = groups if split == "train" else max(4, groups//5)
        for i in range(n):
            topology = topologies[i%len(topologies)]
            state = make_state(topology, int(digest([seed, split, i])[:12],16))
            group = digest(["lic", state])[:24]
            changed = deepcopy(state)
            # Changes genuinely alter the best action, rather than rewarding invariance by accident.
            base_best = optimum(state)[0]
            found = False
            for node in range(8):
                changed = deepcopy(state); changed["benefit"][node] += 100
                if utility(changed, base_best) < optimum(changed)[1]: found = True; break
            if not found:
                changed = deepcopy(state)
                for node in base_best["targets"]: changed["fatigue"][node] += 1000
            if utility(changed, base_best) == optimum(changed)[1]:
                raise ValueError("Counterfactual policy did not change optimal behavior")
            rewired = deepcopy(state)
            rewired["edges"] = sorted(sorted(((a+3)%8,(b+3)%8)) for a,b in state["edges"])
            degree = [0]*8
            for a,b in state["edges"]: degree[a]+=1; degree[b]+=1
            frozen_order = sorted(range(8),key=lambda j:(-degree[j],j))
            for view, semantic, current in [("base","base",state),("paraphrase","base",state),
                                            ("counterfactual","changed",changed),("rewire","rewire",rewired)]:
                for lang in languages:
                    prefix = "CURRENT SNAPSHOT\n" if view == "paraphrase" else ""
                    rows.append({"id": digest([group,view,lang])[:24], "group_id": group,
                      "semantic_id": group+":"+semantic, "family": "lic_"+topology,
                      "split": split, "language": lang, "view": view, "kind": "lic", "state": current,
                      "frozen_order": frozen_order,
                      "prompt": [{"role":"system","content": instruction(lang)+"\n"+OBJECTIVE},
                                 {"role":"user","content": prefix+canonical(current)}]})
        write_jsonl(root/f"{split}.jsonl", rows)
        manifest["splits"][split] = {"rows":len(rows),"groups":n}
    write_json(root/"manifest.json", manifest)
    return manifest


def gold_policy(row):
    return optimum(row["state"])[0]


def verify_policy(row, obj):
    try:
        score = utility(row["state"], obj); best = optimum(row["state"])[1]
        return {"correct": score == best, "reason": "verified" if score == best else "positive_regret",
                "utility": float(score), "optimal_utility": float(best), "regret": float(best-score)}
    except (ValueError, TypeError):
        return {"correct": False, "reason": "invalid_policy"}


def baseline_report(rows):
    records = []
    for row in rows:
        if row["kind"] != "lic" or row["language"] != "en": continue
        for name in ("none", "frozen_degree", "adaptive_degree", "adaptive_greedy", "oracle"):
            if name == "oracle": p = gold_policy(row)
            elif name == "frozen_degree": p = heuristic(row["state"], "degree", row["frozen_order"])
            else: p = heuristic(row["state"], {"none":"none","adaptive_degree":"degree","adaptive_greedy":"greedy"}[name])
            records.append({"id":row["id"],"group_id":row["group_id"],"view":row["view"],"baseline":name,
                            "policy":p,**verify_policy(row,p)})
    return records
