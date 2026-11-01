"""Language-neutral worked derivations for SFT targets ("derivation-first" completions).

The derivation is a formula trace with exact rationals, placed in `explanation` and emitted
BEFORE `status` and `answer`, so an autoregressive student computes before it commits.
The verifier ignores `explanation` and accepts any key order, so evaluation is unchanged.
"""
import json
from fractions import Fraction as F
from .worlds import number, rational, solve
from .languages import EXPLANATIONS, index


def _r(x):
    return rational(F(x))


def _p(x):
    s = _r(x)
    return f"({s})" if "/" in s or s.startswith("-") else s


def _q(p):
    return {k: number(v) for k, v in p.items() if v is not None and k not in ("domain", "convention")}


def derivation(world):
    f, p = world["family"], world["facts"]
    status, _ = solve(world)
    missing = [k for k, v in p.items() if v is None]
    if status == "needs_clarification":
        if missing: return f"missing facts: {', '.join(missing)} -> needs_clarification"
        if f == "variance": return "convention is neither population nor sample -> needs_clarification"
        if f == "product": return "domain is neither nonnegative nor real -> needs_clarification"
        if f == "affine": return "a = 0 and b = 0 -> every x satisfies 0*x = 0 -> needs_clarification"
        return "underdetermined -> needs_clarification"
    q = _q(p)
    if status == "infeasible":
        if f == "rate": return f"duration_h = {_r(q['duration_h'])} <= 0 -> infeasible"
        if f == "affine": return f"a = 0 and b = {_r(q['b'])} != 0 -> 0*x = {_r(q['b'])} has no solution -> infeasible"
        if f == "product": return f"total = {_r(q['total'])} < 0 with x,y >= 0 -> infeasible"
        if f == "energy": return f"watts = {_r(q['watts'])} <= 0 -> infeasible"
        if f == "workrate": return f"a time is <= 0 ({_r(q['hours_a'])}, {_r(q['hours_b'])}) -> infeasible"
        if f == "rectangle": return f"perimeter = {_r(q['perimeter'])} < 0 -> infeasible"
        if f == "journey": return f"a speed is <= 0 ({_r(q['speed_a'])}, {_r(q['speed_b'])}) -> infeasible"
        if f == "production": return f"x+y >= {_r(q['minimum'])} with 2x+3y <= {_r(q['capacity'])} has no nonnegative integer solution -> infeasible"
        return "constraints admit no solution -> infeasible"
    if status == "unbounded":
        return "domain = real: with y -> -inf and x = total - y, x*x*y -> +inf -> unbounded"
    # solved
    if f == "rate":
        return f"speed_kmh = distance_km/duration_h = {_p(q['distance_km'])}/{_p(q['duration_h'])} = {_r(q['distance_km']/q['duration_h'])}"
    if f == "affine":
        return f"x = b/a = {_p(q['b'])}/{_p(q['a'])} = {_r(q['b']/q['a'])}"
    if f == "mean":
        num = q["a"]*q["weight_a"]+q["b"]*q["weight_b"]; den = q["weight_a"]+q["weight_b"]
        return (f"mean = (a*weight_a + b*weight_b)/(weight_a + weight_b) = "
                f"({_p(q['a'])}*{_p(q['weight_a'])} + {_p(q['b'])}*{_p(q['weight_b'])})/({_p(q['weight_a'])} + {_p(q['weight_b'])}) = "
                f"{_p(num)}/{_p(den)} = {_r(num/den)}")
    if f == "variance":
        v = [q["a"], q["b"], q["c"]]; m = sum(v)/3; d = [(x-m)**2 for x in v]; n = 3 if p["convention"] == "population" else 2
        return (f"mean = ({_r(v[0])} + {_r(v[1])} + {_r(v[2])})/3 = {_r(m)}; squared deviations = "
                f"{_r(d[0])}, {_r(d[1])}, {_r(d[2])}; sum = {_r(sum(d))}; {p['convention']} divides by {n}: "
                f"variance = {_p(sum(d))}/{n} = {_r(sum(d)/n)}")
    if f == "product":
        x, y = 2*q["total"]/3, q["total"]/3
        return (f"maximize x*x*y with x+y = {_r(q['total'])}, x,y >= 0: optimum at x = 2*total/3 = {_r(x)}, y = total/3 = {_r(y)}; "
                f"maximum = x*x*y = {_p(x)}*{_p(x)}*{_p(y)} = {_r(x*x*y)}")
    if f == "energy":
        usable = q["capacity_wh"]*q["efficiency"]
        return (f"usable_wh = capacity_wh*efficiency = {_p(q['capacity_wh'])}*{_p(q['efficiency'])} = {_r(usable)}; "
                f"hours = usable_wh/watts = {_p(usable)}/{_p(q['watts'])} = {_r(usable/q['watts'])}")
    if f == "workrate":
        a, b = q["hours_a"], q["hours_b"]
        return (f"rate = 1/hours_a + 1/hours_b = 1/{_r(a)} + 1/{_r(b)} = {_r(1/a+1/b)}; "
                f"hours = 1/rate = hours_a*hours_b/(hours_a+hours_b) = {_p(a*b)}/{_p(a+b)} = {_r(1/(1/a+1/b))}")
    if f == "rectangle":
        s = q["perimeter"]/4
        return f"side = perimeter/4 = {_r(q['perimeter'])}/4 = {_r(s)}; maximum_area = side*side = {_p(s)}*{_p(s)} = {_r(s*s)}"
    if f == "system":
        s, d = q["sum"], q["difference"]
        return (f"x = (sum + difference)/2 = ({_r(s)} + {_r(d)})/2 = {_r((s+d)/2)}; "
                f"y = (sum - difference)/2 = ({_r(s)} - {_r(d)})/2 = {_r((s-d)/2)}")
    if f == "journey":
        ta, tb = q["distance_a"]/q["speed_a"], q["distance_b"]/q["speed_b"]
        return (f"time_a = {_p(q['distance_a'])}/{_p(q['speed_a'])} = {_r(ta)}; time_b = {_p(q['distance_b'])}/{_p(q['speed_b'])} = {_r(tb)}; "
                f"total_time = {_r(ta+tb)}; speed = (distance_a + distance_b)/total_time = "
                f"{_p(q['distance_a']+q['distance_b'])}/{_p(ta+tb)} = {_r((q['distance_a']+q['distance_b'])/(ta+tb))}")
    if f == "production":
        cap = int(q["capacity"]); px, py = q["profit_x"], q["profit_y"]
        _, ans = solve(world)
        return (f"profit per unit of capacity: x gives {_r(px)}/2 = {_r(px/2)}, y gives {_r(py)}/3 = {_r(py/3)}; "
                f"check integer points with 2x+3y <= {cap} and x+y >= {_r(q['minimum'])}, including leftover-capacity mixes; "
                f"best: x = {ans['x']}, y = {ans['y']}, profit = {_p(px)}*{ans['x']} + {_p(py)}*{ans['y']} = {ans['profit']}")
    if f == "interest":
        P, r, n = q["principal"], q["rate"], int(q["years"]); g = 1+r; steps = []; v = P
        for i in range(1, n+1):
            v = v*g; steps.append(f"year {i}: {_p(v/g)}*{_p(g)} = {_r(v)}")
        return f"growth factor = 1 + rate = {_r(g)}; " + "; ".join(steps) + f"; balance = {_r(v)}"
    raise ValueError(f"Unknown family: {f}")


def worked_completion(row):
    """Ordered JSON: facts -> explanation (derivation + authored sentence) -> status -> answer."""
    from .dataset import gold
    g = gold(row)
    text = derivation(row["world"]) + ". " + EXPLANATIONS[g["status"]][index(row["language"])]
    ordered = {"facts": g["facts"], "explanation": text, "status": g["status"], "answer": g["answer"]}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
