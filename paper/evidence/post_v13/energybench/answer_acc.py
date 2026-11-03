"""Answer-accuracy secondary score for the energy-reasoning benchmark.
A run is correct if its text contains the reference key value (within 1% relative, or 0.01 absolute),
where the key value is the first number in the reference solution that does not already appear in the problem.
Numbers are parsed with thousands/decimal separators of either convention (1,000 / 1.000 / 2,5 / 2.5)."""
import json, os, re, sys, csv
B = "/gpfs/scratch/bsc21/bsc021885/cf/energybench"
L = ["en", "es", "ca", "ar", "lb", "eu", "fr", "pt", "it", "de", "ru", "zh-cn", "zh-tw"]
TOK = re.compile(r"-?\d[\d.,   ]*\d|-?\d")

def vals(text, groups=False):
    out = []
    for m in TOK.finditer(text):
        s = re.sub(r"[   ]", "", m.group(0)).strip(".,")
        cands = set()
        if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s): cands.add(re.sub(r"[.,]", "", s))   # thousands
        if re.fullmatch(r"-?\d+[.,]\d+", s): cands.add(s.replace(",", "."))             # decimal
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+\.\d+", s): cands.add(s.replace(",", ""))   # 1,234.5
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+,\d+", s): cands.add(s.replace(".", "").replace(",", "."))
        if re.fullmatch(r"-?\d+", s): cands.add(s)
        fl = []
        for c in cands:
            try: fl.append(float(c))
            except ValueError: pass
        if not fl: continue
        if groups: out.append(fl)
        else: out.extend(fl)
    return out

def close(a, b): return abs(a - b) <= max(0.01 * abs(b), 0.011)

def key_value(problem, solution):
    pv = vals(problem)
    for g in vals(solution, groups=True):
        v = max(g, key=abs)  # ambiguous '2,500' / '1.000' -> thousands reading
        if not any(close(v, p) for p in pv): return v
    return None

rows = []
for od in ("out", "out_chat"):
    for s in sorted(os.listdir(f"{B}/{od}")):
        acc = {}
        for l in L:
            f = f"{B}/{od}/{s}/{l}.json"
            if not os.path.exists(f): continue
            d = json.load(open(f)); probs = d["problems"]; sols = d["solutions"]
            sols = list(sols.values()) if isinstance(sols, dict) else sols
            n = c = 0
            for p, runs in zip(probs, sols):
                kv = key_value(p["problem"], p["solution"])
                if kv is None or not runs: continue
                for r in runs:
                    n += 1; c += any(close(v, kv) for v in vals(r))
            acc[l] = c / n if n else None
        if len(acc) == 13: rows.append([od, s, round(sum(acc.values()) / 13, 3)] + [round(acc[l], 3) for l in L])
with open(f"{B}/answer_accuracy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["protocol", "slot", "mean"] + L); w.writerows(rows)
for r in rows: print(f"{r[0]:8s} {r[1]:22s} {r[2]:.3f}")
