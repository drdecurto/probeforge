"""Significance summary across all arms (writes $S/sig/*.csv).
main: contrastforge compare (paired group bootstrap, 12 held-out groups) of each arm vs its base.
energy: paired cluster bootstrap over the 79 problems (13 languages each) of semantic score and answer accuracy, raw and chat.
mgsm: paired cluster bootstrap over the 250 MGSM items (6 languages each), strict-match.
belebele/gmmlu: two-proportion normal interval on the macro-averaged accuracy (no per-item logs)."""
import json, os, glob, csv, random, subprocess, math
S = "/gpfs/scratch/bsc21/bsc021885/cf"; O = f"{S}/sig"; os.makedirs(O, exist_ok=True)
R = lambda p: f"{S}/{p}"
FAM = {
 "Qwen3-8B": ("evalv2/report12_base.json", ["evalv2/report12_translated.json", "evalv2/report12_random.json", "evalv2/report12_random_s1.json", "evalv2/report12_random_s2.json",
              "evalv2/report12_mined.json", "evalv2/report12_mined_s1.json", "evalv2/report12_mined_s2.json", "evalv2/report12_no_counterfactual.json", "evalv2/report12_dpo_mined.json",
              "evalv2/report12_dpo_onp.json", "evalv2/report12_v3_random.json", "evalv2/report12_v3_mined.json", "evalv2/report12_xfer5_random.json", "evalv2/report12_xfer5_mined.json",
              "evalv4/report12_v4_8b.json", "evalv4b/report12_v4b_8b.json", "evalv4c/report12_8b_s1.json", "evalv4c/report12_8b_s2.json", "evalv4c/report12_8b_s3.json"]),
 "Qwen3-14B": ("eval14b/report12_base14b.json", sorted(f for f in glob.glob(R("eval14b/report12_*.json")) if "base14b" not in f) + [R(f"evalv4c/report12_14b_s{i}.json") for i in (1, 2, 3)]),
 "Phi-4": ("evalphi4/report12_base.json", [f"evalphi4/report12_{a}.json" for a in ("translated", "random", "mined")] + [f"evalphi4s/report12_{a}_s{i}.json" for a in ("random", "mined") for i in (1, 2)]),
 "Phi-4-mini": ("evalphi4mini/report12_base.json", [f"evalphi4mini/report12_{a}.json" for a in ("translated", "random", "mined")] + ["evalv4/report12_v4_phi4mini.json", "evalv4b/report12_v4b_phi4mini.json"] + [f"evalv4c/report12_phi4mini_s{i}.json" for i in (1, 2, 3)]),
 "Aya-Expanse-8B": ("evalv4c/report12_aya_base.json", ["evalv4c/report12_aya_s1.json"] + sorted(glob.glob(R("evalaya/report12_*.json"))) + glob.glob(R("evalv4c/report12_aya_own_s1.json"))),
}
rows = []
for fam, (base, arms) in FAM.items():
    arms = [a if a.startswith("/") else R(a) for a in arms]; arms = [a for a in arms if os.path.exists(a) and not a.endswith("report12_base.json")]
    out = f"{O}/cmp_{fam}.json"
    subprocess.run(["contrastforge", "compare", "--reports", R(base), *arms, "--out", out], check=True, stdout=subprocess.DEVNULL)
    for x in json.load(open(out))["comparisons"]:
        t = x["treatment"]; b = json.load(open(t))["budgets"]; b = b.get("512") or next(iter(b.values()))
        rows.append([fam, t.replace(S + "/", ""), round(b["pass_at_1"], 3), round(x["mean_paired_group_difference"], 3), *[round(v, 3) for v in x["paired_ci95"]], "yes" if x["paired_ci95"][0] > 0 else ("neg" if x["paired_ci95"][1] < 0 else "no")])
with open(f"{O}/sig_main.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "arm", "pass_at_1", "delta_vs_base", "ci_lo", "ci_hi", "significant"]); w.writerows(rows)
print("main", len(rows))

# ---------- energy benchmark
B = f"{S}/energybench"; L = ["en", "es", "ca", "ar", "lb", "eu", "fr", "pt", "it", "de", "ru", "zh-cn", "zh-tw"]
ns = {}; exec(open(f"{B}/answer_acc.py").read().split("rows = []")[0], ns)
def items(od, s):
    sem, acc = {}, {}
    for l in L:
        f = f"{B}/{od}/{s}/{l}.json"
        if not os.path.exists(f): return None
        d = json.load(open(f)); a = d["average_scores"]; a = list(a.values()) if isinstance(a, dict) else a
        sols = d["solutions"]; sols = list(sols.values()) if isinstance(sols, dict) else sols
        for i, (p, runs, x) in enumerate(zip(d["problems"], sols, a)):
            if x is not None: sem[(l, i)] = x
            kv = ns["key_value"](p["problem"], p["solution"])
            if kv is not None and runs: acc[(l, i)] = sum(any(ns["close"](v, kv) for v in ns["vals"](r)) for r in runs) / len(runs)
    return sem, acc
def boot(A, Bv, n=2000, seed=1):
    keys = [k for k in A if k in Bv]; cl = sorted({k[1] for k in keys}); by = {c: [Bv[k] - A[k] for k in keys if k[1] == c] for c in cl}
    m = sum(Bv[k] - A[k] for k in keys) / len(keys); rng = random.Random(seed); bs = []
    for _ in range(n):
        smp = [x for c in (rng.choice(cl) for _ in cl) for x in by[c]]; bs.append(sum(smp) / len(smp))
    bs.sort(); return m, bs[int(.025 * n)], bs[int(.975 * n)]
EB = {"Qwen3-8B": ("qwen3-8b-base", ["qwen3-8b-mined", "qwen3-8b-v3random", "qwen3-8b-v4", "qwen3-8b-v4b", "v4c-8b_s1", "v4c-8b_s2", "v4c-8b_s3"]),
      "Qwen3-14B": ("qwen3-14b-base", ["qwen3-14b-random", "qwen3-14b-mined", "qwen3-14b-v3mined", "qwen3-14b-distill", "qwen3-14b-mixed50", "v4c-14b_s1", "v4c-14b_s2", "v4c-14b_s3"]),
      "Phi-4": ("phi-4-base", ["phi-4-mined", "phi-4-mined_s1", "phi-4-mined_s2"]),
      "Phi-4-mini": ("phi-4-mini-instruct", ["phi-4-mini-mined", "phi-4-mini-random", "phi-4-mini-v4", "phi-4-mini-v4b", "v4c-phi4mini_s1", "v4c-phi4mini_s2", "v4c-phi4mini_s3"]),
      "Aya-Expanse-8B": ("aya-expanse-8b", ["v4c-aya_s1", "aya-mined", "v4c-aya_own_s1"])}
erows = []
for fam, (base, arms) in EB.items():
    for od in ("out", "out_chat"):
        bi = items(od, base)
        if bi is None: continue
        for a in arms:
            ai = items(od, a)
            if ai is None: continue
            for k, name in ((0, "semantic"), (1, "answer_acc")):
                m, lo, hi = boot(bi[k], ai[k])
                erows.append([fam, a, "raw" if od == "out" else "chat", name, round(sum(ai[k].values()) / len(ai[k]), 3), round(m, 3), round(lo, 3), round(hi, 3), "yes" if lo > 0 else ("neg" if hi < 0 else "no")])
with open(f"{O}/sig_energy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "arm", "protocol", "metric", "score", "delta_vs_base", "ci_lo", "ci_hi", "significant"]); w.writerows(erows)
print("energy", len(erows))

# ---------- MGSM (paired, strict) + Belebele/GMMLU (two-proportion)
SD = f"{S}/stdbench"
def mgsm(name):
    o = {}
    for f in glob.glob(f"{SD}/out/{name}/mgsm/**/samples_mgsm_native_cot_*.jsonl", recursive=True):
        lang = f.split("samples_mgsm_native_cot_")[1].split("_")[0]
        for l in open(f):
            r = json.loads(l)
            if r.get("filter") == "strict-match": o[(lang, r["doc_id"])] = float(r["exact_match"])
    return o if len(o) >= 1500 else None
summ = {r["model"]: r for r in csv.DictReader(open(f"{SD}/stdbench_summary.csv"))}
SB = {"Qwen3-8B": ("qwen3-8b-base", ["qwen3-8b-mined", "qwen3-8b-v3random", "qwen3-8b-v4", "qwen3-8b-v4b", "v4c-8b_s1", "v4c-8b_s2", "v4c-8b_s3"]),
      "Qwen3-14B": ("qwen3-14b-base", ["qwen3-14b-random", "qwen3-14b-mined", "qwen3-14b-v3mined", "qwen3-14b-distill", "qwen3-14b-mixed50", "v4c-14b_s1", "v4c-14b_s2", "v4c-14b_s3"]),
      "Phi-4": ("phi-4-base", ["phi-4-mined", "phi-4-mined_s1", "phi-4-mined_s2"]),
      "Phi-4-mini": ("phi-4-mini-instruct", ["phi-4-mini-mined", "phi-4-mini-random", "phi-4-mini-v4", "phi-4-mini-v4b", "v4c-phi4mini_s1", "v4c-phi4mini_s2", "v4c-phi4mini_s3"]),
      "Aya-Expanse-8B": ("aya-expanse-8b", ["v4c-aya_s1", "aya-mined", "v4c-aya_own_s1"])}
srows = []
for fam, (base, arms) in SB.items():
    bm = mgsm(base)
    for a in arms:
        if a not in summ: continue
        am = mgsm(a)
        if bm and am:
            m, lo, hi = boot(bm, am, seed=2); srows.append([fam, a, "mgsm_strict", round(sum(am.values()) / len(am), 3), round(m, 3), round(lo, 3), round(hi, 3), "yes" if lo > 0 else ("neg" if hi < 0 else "no")])
        for col, n, name in (("bele_mean", 12 * 900, "belebele"), ("gmmlu_mean", 8 * 400, "global_mmlu_lite")):
            try: pa, pb = float(summ[a][col]), float(summ[base][col])
            except (KeyError, ValueError, TypeError): continue
            se = math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n); d = pa - pb
            srows.append([fam, a, name, round(pa, 3), round(d, 3), round(d - 1.96 * se, 3), round(d + 1.96 * se, 3), "yes" if d - 1.96 * se > 0 else ("neg" if d + 1.96 * se < 0 else "no")])
with open(f"{O}/sig_standard.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "arm", "benchmark", "score", "delta_vs_base", "ci_lo", "ci_hi", "significant"]); w.writerows(srows)
print("standard", len(srows))
