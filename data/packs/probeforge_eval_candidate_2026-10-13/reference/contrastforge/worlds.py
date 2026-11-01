"""Small exact problem generators. No LLM or arbitrary code is used as an oracle."""
from copy import deepcopy
from fractions import Fraction as F
import random

FAMILIES = {
    "train": ("rate", "affine", "mean", "variance", "product", "energy"),
    "validation": ("workrate", "rectangle", "system"),
    "test": ("journey", "production", "interest"),
}
PRIMARY = dict(rate="distance_km", affine="b", mean="a", variance="a",
               product="total", energy="capacity_wh", workrate="hours_a",
               rectangle="perimeter", system="sum", journey="distance_a",
               production="capacity", interest="principal")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, F)):
        raise ValueError("Expected a finite number or rational string")
    if len(str(value)) > 80:
        raise ValueError("Number too long")
    x = F(str(value))
    if abs(x.numerator) > 10**30 or x.denominator > 10**20:
        raise ValueError("Number outside supported limits")
    return x


def rational(x):
    x = number(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def solve(world):
    """Return semantic status and exact answer, independent of surface language."""
    f, p = world["family"], world["facts"]
    if any(x is None for x in p.values()):
        return "needs_clarification", {}
    q = {k: number(v) for k, v in p.items() if k not in ("domain", "convention")}
    status, out = "solved", {}
    if f == "rate":
        if q["duration_h"] <= 0: return "infeasible", {}
        out = {"speed_kmh": q["distance_km"] / q["duration_h"]}
    elif f == "affine":
        if q["a"] == 0:
            return ("needs_clarification" if q["b"] == 0 else "infeasible"), {}
        out = {"x": q["b"] / q["a"]}
    elif f == "mean":
        out = {"mean": (q["a"]*q["weight_a"]+q["b"]*q["weight_b"])/(q["weight_a"]+q["weight_b"])}
    elif f == "variance":
        values = [q[k] for k in ("a", "b", "c")]
        mean = sum(values)/3
        if p["convention"] not in ("population", "sample"): return "needs_clarification", {}
        out = {"variance": sum((x-mean)**2 for x in values)/(3 if p["convention"] == "population" else 2)}
    elif f == "product":
        if p["domain"] == "real": return "unbounded", {}
        if p["domain"] != "nonnegative": return "needs_clarification", {}
        if q["total"] < 0: return "infeasible", {}
        x, y = 2*q["total"]/3, q["total"]/3
        out = {"x": x, "y": y, "maximum": x*x*y}
    elif f == "energy":
        if q["watts"] <= 0: return "infeasible", {}
        out = {"hours": q["capacity_wh"]*q["efficiency"] / q["watts"]}
    elif f == "workrate":
        if min(q.values()) <= 0: return "infeasible", {}
        out = {"hours": 1/(1/q["hours_a"]+1/q["hours_b"])}
    elif f == "rectangle":
        if q["perimeter"] < 0: return "infeasible", {}
        out = {"side": q["perimeter"]/4, "maximum_area": (q["perimeter"]/4)**2}
    elif f == "system":
        out = {"x": (q["sum"]+q["difference"])/2, "y": (q["sum"]-q["difference"])/2}
    elif f == "journey":
        if min(q["speed_a"], q["speed_b"]) <= 0: return "infeasible", {}
        out = {"speed": (q["distance_a"]+q["distance_b"])/(q["distance_a"]/q["speed_a"]+q["distance_b"]/q["speed_b"])}
    elif f == "production":
        # Exact integer enumeration, with an explicit minimum-production constraint.
        cap = int(q["capacity"])
        feasible = [(q["profit_x"]*x+q["profit_y"]*y, x, y)
                    for x in range(cap//2+1) for y in range(cap//3+1)
                    if 2*x+3*y <= cap and x+y >= q["minimum"]]
        if not feasible: return "infeasible", {}
        # Public tie rule: maximize profit, then minimize x, then minimize y.
        best = min(feasible, key=lambda z: (-z[0], z[1], z[2]))
        out = {"profit": best[0], "x": best[1], "y": best[2]}
    elif f == "interest":
        out = {"balance": q["principal"]*(1+q["rate"])**int(q["years"])}
    else:
        raise ValueError(f"Unknown family: {f}")
    return status, {k: rational(v) for k, v in out.items()}


def make_world(family, seed):
    r = random.Random(seed)
    a, b, c = r.randint(2, 2000), r.randint(2, 50), r.randint(3, 80)
    facts = {
        "rate": dict(distance_km=a*10, duration_h=b),
        "affine": dict(a=b, b=a*c),
        "mean": dict(a=a, b=b, weight_a=c, weight_b=2),
        "variance": dict(a=a, b=b, c=c, convention="population"),
        "product": dict(total=a, domain="nonnegative"),
        "energy": dict(capacity_wh=a*10, watts=b, efficiency="4/5"),
        "workrate": dict(hours_a=a, hours_b=b),
        "rectangle": dict(perimeter=a*4),
        "system": dict(sum=a+b, difference=a-b),
        "journey": dict(distance_a=a*10, distance_b=b*10, speed_a=30, speed_b=60),
        "production": dict(capacity=a%60+20, profit_x=b, profit_y=c, minimum=0),
        "interest": dict(principal=a*100, rate="1/20", years=min(b, 4)),
    }[family]
    return {"family": family, "facts": facts}


def variants(world, metamorphic=False):
    """Equivalent views share world semantics; changed views have different gold answers."""
    base = deepcopy(world)
    changed = deepcopy(world)
    key = PRIMARY[world["family"]]
    # Search a bounded set to avoid accidental answer-preserving 'counterfactuals'.
    for delta in (1, 3, 7, 13):
        changed["facts"][key] = rational(number(base["facts"][key])+delta)
        if solve(changed) != solve(base): break
    else:
        raise ValueError("Could not construct a meaningful counterfactual")
    boundary = deepcopy(world)
    f = world["family"]
    if f == "product": boundary["facts"]["domain"] = "real"
    elif f == "affine": boundary["facts"]["a"] = 0
    elif f == "variance": boundary["facts"]["convention"] = None
    elif f == "production": boundary["facts"]["minimum"] = 1000
    elif f == "rectangle": boundary["facts"]["perimeter"] = -1
    else: boundary["facts"][key] = None
    result = [("base", "base", base), ("paraphrase", "base", deepcopy(base)),
              ("distractor", "base", deepcopy(base)), ("counterfactual", "changed", changed),
              ("boundary", "boundary", boundary)]
    if f in ("rate", "energy"):
        result.append(("units", "base", deepcopy(base)))
    if metamorphic:
        # Metamorphic relations (context injection, expansion): answer-preserving views.
        for view in ("context_academic", "context_business", "expansion"):
            result.append((view, "base", deepcopy(base)))
    return result
