"""Regenerate paper figures and tables from the archived evaluation reports in data/.

usage: python make_figures.py   (run from the paper/ folder; writes figures/*.png|pdf and tables/*.csv)
All numbers come from contrastforge `evaluate` reports (budgets["512"]) copied from the MN5 runs;
thinking-mode numbers (small subsets, read from the chain logs) are typed in THINK below.
"""
import json, csv, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("data"); F = Path("figures"); T = Path("tables"); F.mkdir(exist_ok=True); T.mkdir(exist_ok=True)
C = {"base": "#8a8985", "translated": "#1baf7a", "random": "#2a78d6", "mined": "#eb6834", "v3": "#eda100", "mixed": "#e87ba4", "ref": "#52514e"}
LANGS = ["en", "es", "ca", "ar", "lb", "eu", "fr", "pt", "it", "de", "ru", "zh-cn", "zh-tw"]
VIEWS = ["base", "paraphrase", "distractor", "context_academic", "context_business", "expansion", "boundary", "counterfactual"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})

def rep(path):
    b = json.load(open(D/path))["budgets"]
    return b["512"] if "512" in b else next(iter(b.values()))  # Muse reports use a 4,096-token budget

# ---------- main benchmark (test_eval12: 12 held-out groups x 13 languages x 8 views = 1,248 prompts)
R8 = {"base": ["round4/report12_base.json"], "translated": ["round4/report12_translated.json"],
      "random": ["round4/report12_random.json", "round4/report12_random_s1.json", "round4/report12_random_s2.json"],
      "mined": ["round4/report12_mined.json", "round4/report12_mined_s1.json", "round4/report12_mined_s2.json"]}
R14 = {"base": ["round6/report12_base14b.json"], "translated": ["round6/report12_translated.json"],
       "random": ["round6/report12_random.json", "round6/report12_random_s1.json", "round6/report12_random_s2.json"],
       "mined": ["round6/report12_mined.json", "round6/report12_mined_s1.json", "round6/report12_mined_s2.json"]}
RPHI = {"base": ["round8/report12_base.json"], "translated": ["round8/report12_translated.json"],
        "random": ["round8/report12_random.json", "round19/report12_phi4_random_s1.json", "round19/report12_phi4_random_s2.json"],
        "mined": ["round8/report12_mined.json", "round19/report12_phi4_mined_s1.json", "round19/report12_phi4_mined_s2.json"]}
V3 = {"random": ["round9/report12_v3_random.json", "round10/report12_v3_random_s2.json", "round10/report12_v3_random_s3.json"],
      "mined": ["round9/report12_v3_mined.json", "round10/report12_v3_mined_s2.json", "round10/report12_v3_mined_s3.json"]}

def agg(paths, key="pass_at_1"):
    v = [rep(p)[key] for p in paths]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), v

rows = []
for model, R in (("Qwen3-8B", R8), ("Qwen3-14B", R14), ("Phi-4", RPHI)):
    for arm, paths in R.items():
        m, s, v = agg(paths); w = agg(paths, "worst_language")[0]
        b = st.mean(rep(p)["per_view"]["boundary"] for p in paths); cf = st.mean(rep(p)["per_view"]["counterfactual"] for p in paths)
        tok = st.mean(rep(p)["mean_completion_tokens"] for p in paths)
        rows.append(dict(model=model, arm=arm, seeds=len(v), pass_at_1=round(m, 3), sd=round(s, 3), worst_language=round(w, 3), boundary=round(b, 3), counterfactual=round(cf, 3), tokens=round(tok)))
for arm, paths in V3.items():
    m, s, v = agg(paths); w = agg(paths, "worst_language")[0]; b = st.mean(rep(p)["per_view"]["boundary"] for p in paths)
    rows.append(dict(model="Qwen3-14B", arm=f"v3 {arm}", seeds=len(v), pass_at_1=round(m, 3), sd=round(s, 3), worst_language=round(w, 3), boundary=round(b, 3),
                     counterfactual=round(st.mean(rep(p)["per_view"]["counterfactual"] for p in paths), 3), tokens=round(st.mean(rep(p)["mean_completion_tokens"] for p in paths))))
for model, arm, path in (("Phi-4-mini", "base", "round15/evalphi4mini_report12_base.json"), ("Phi-4-mini", "translated", "round15/evalphi4mini_report12_translated.json"),
                         ("Phi-4-mini", "random", "round15/evalphi4mini_report12_random.json"), ("Phi-4-mini", "mined", "round15/evalphi4mini_report12_mined.json"),
                         ("Qwen3-8B", "v4 (JSON+prose+GSM8K)", "round15/evalv4_report12_v4_8b.json"), ("Phi-4-mini", "v4 (JSON+prose+GSM8K)", "round15/evalv4_report12_v4_phi4mini.json"),
                         ("Qwen3-8B", "v4b (concise teacher)", "round15/evalv4b_report12_v4b_8b.json"), ("Phi-4-mini", "v4b (concise teacher)", "round15/evalv4b_report12_v4b_phi4mini.json")):
    b = rep(path); rows.append(dict(model=model, arm=arm, seeds=1, pass_at_1=round(b["pass_at_1"], 3), sd=0.0, worst_language=round(b["worst_language"], 3),
                                    boundary=round(b["per_view"]["boundary"], 3), counterfactual=round(b["per_view"]["counterfactual"], 3), tokens=round(b["mean_completion_tokens"])))
for arm, path in (("frontier reference, 4k budget", "round14/report12_muse.json"), ("frontier reference, 512 budget", "round14/report12_muse_strict512.json")):
    b = rep(path); rows.append(dict(model="Muse-Spark-Open-816B-A42B", arm=arm, seeds=1, pass_at_1=round(b["pass_at_1"], 3), sd=0.0, worst_language=round(b["worst_language"], 3),
                                    boundary=round(b["per_view"]["boundary"], 3), counterfactual=round(b["per_view"]["counterfactual"], 3), tokens=round(b["mean_completion_tokens"])))
with open(T/"table1_main_benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# Figure 1: arms by model size
fig, ax = plt.subplots(figsize=(6.4, 3.2))
models = ["Qwen3-8B", "Qwen3-14B", "Phi-4"]; arms = ["base", "translated", "random", "mined"]; wd = 0.19
for j, arm in enumerate(arms):
    xs = [i + (j - 1.5) * wd for i in range(len(models))]
    ys = [next(r["pass_at_1"] for r in rows if r["model"] == m and r["arm"] == arm) for m in models]
    es = [next(r["sd"] for r in rows if r["model"] == m and r["arm"] == arm) for m in models]
    ax.bar(xs, ys, wd - 0.02, color=C[arm], label=arm, yerr=es, capsize=2, error_kw=dict(lw=0.8, color="#52514e"))
    for x, y, e in zip(xs, ys, es): ax.text(x, y + e + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_ylim(0, 0.52); ax.set_ylabel("exact pass@1 (13 languages, 12 held-out groups)")
ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left"); ax.set_title("Verified-contrast SFT by base model and curriculum arm (error bars: seed sd)", fontsize=9)
fig.tight_layout(); fig.savefig(F/"fig1_arms_by_model.png"); fig.savefig(F/"fig1_arms_by_model.pdf"); plt.close(fig)

# Figure 2: per-language (seed means)
def per_lang(paths):
    return {l: st.mean(rep(p)["per_language"][l] for p in paths) for l in LANGS}
series = [("Qwen3-8B base", per_lang(R8["base"]), C["base"], "o"), ("Qwen3-8B mined (3 seeds)", per_lang(R8["mined"]), C["mined"], "o"),
          ("Qwen3-14B base", per_lang(R14["base"]), C["base"], "s"), ("Qwen3-14B random (3 seeds)", per_lang(R14["random"]), C["random"], "s"),
          ("Phi-4 mined (3 seeds)", per_lang(RPHI["mined"]), C["v3"], "^")]
with open(T/"table2_per_language.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series"] + LANGS)
    for name, d, _, _ in series: w.writerow([name] + [round(d[l], 3) for l in LANGS])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for name, d, col, mk in series:
    ax.plot(range(len(LANGS)), [d[l] for l in LANGS], marker=mk, ms=4.5, lw=1.2, color=col, label=name, alpha=0.95 if "base" not in name else 0.7, ls="-" if "base" not in name else "--")
ax.set_xticks(range(len(LANGS))); ax.set_xticklabels(LANGS); ax.set_ylim(0, 0.55); ax.set_ylabel("pass@1"); ax.legend(frameon=False, fontsize=7.5, ncol=2)
ax.set_title("Per-language accuracy on the main benchmark", fontsize=9); fig.tight_layout(); fig.savefig(F/"fig2_per_language.png"); fig.savefig(F/"fig2_per_language.pdf"); plt.close(fig)

# Figure 3: per-view (invariance vs sensitivity) + diag60 counterfactual
pv8 = {v: st.mean(rep(p)["per_view"][v] for p in R8["mined"]) for v in VIEWS}
pv14 = {v: st.mean(rep(p)["per_view"][v] for p in R14["random"]) for v in VIEWS}
pvb14 = rep("round6/report12_base14b.json")["per_view"]; pvb8 = rep("round4/report12_base.json")["per_view"]
d60_8 = rep("round8/report_mined8b.json")["per_view"]; d60_14 = rep("round8/report_random14b.json")["per_view"]
with open(T/"table3_per_view.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series"] + VIEWS)
    for name, d in (("8B base", pvb8), ("8B mined (3 seeds)", pv8), ("14B base", pvb14), ("14B random (3 seeds)", pv14), ("8B mined, diag60", d60_8), ("14B random, diag60", d60_14)):
        w.writerow([name] + [round(d[v], 3) for v in VIEWS])
fig, ax = plt.subplots(figsize=(6.4, 3.0)); x = range(len(VIEWS)); wd = 0.2
ax.bar([i - 1.5 * wd for i in x], [pvb14[v] for v in VIEWS], wd - .02, color=C["base"], label="14B base")
ax.bar([i - 0.5 * wd for i in x], [pv8[v] for v in VIEWS], wd - .02, color=C["mined"], label="8B mined (3 seeds)")
ax.bar([i + 0.5 * wd for i in x], [pv14[v] for v in VIEWS], wd - .02, color=C["random"], label="14B random (3 seeds)")
ax.bar([i + 1.5 * wd for i in x], [d60_14[v] for v in VIEWS], wd - .02, color=C["random"], hatch="////", edgecolor="white", lw=0, label="14B random on diag60 pack")
ax.set_xticks(list(x)); ax.set_xticklabels([v.replace("context_", "ctx. ") for v in VIEWS], rotation=20, ha="right"); ax.set_ylabel("pass@1"); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7.5, ncol=2); ax.set_title("Accuracy by view: answer-preserving views stay flat; status views are learned", fontsize=9)
fig.tight_layout(); fig.savefig(F/"fig3_per_view.png"); fig.savefig(F/"fig3_per_view.pdf"); plt.close(fig)

# Figure 4: confirmation packs (core60 / diag60) with 60-group CIs
packs = [("core60", "8B base", "round7/report_base8b.json", C["base"]), ("core60", "8B mined", "round7/report_mined8b.json", C["mined"]),
         ("core60", "14B base", "round7/report_base14b.json", C["base"]), ("core60", "14B random", "round7/report_random14b.json", C["random"]),
         ("diag60", "8B mined", "round8/report_mined8b.json", C["mined"]), ("diag60", "14B random", "round8/report_random14b.json", C["random"]),
         ("diag60", "14B v3 random", "round9/report_v3random14b.json", C["v3"]),
         ("core60", "Muse-Spark (ref., 4k)", "round14/report_core60_muse.json", C["ref"]), ("diag60", "Muse-Spark (ref., 4k)", "round14/report_diag60_muse.json", C["ref"])]
prow = []
for pack, name, path, col in packs:
    b = rep(path); prow.append(dict(pack=pack, model=name, pass_at_1=round(b["pass_at_1"], 3), ci_lo=round(b["group_mean_accuracy_ci95"][0], 3), ci_hi=round(b["group_mean_accuracy_ci95"][1], 3),
                                    worst_language=round(b["worst_language"], 3), boundary=round(b["per_view"]["boundary"], 3), counterfactual=round(b["per_view"]["counterfactual"], 3), **{f"family_{k}": round(v, 3) for k, v in b["per_family"].items()}))
with open(T/"table4_packs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=prow[0].keys()); w.writeheader(); w.writerows(prow)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, (r, (_, _, _, col)) in enumerate(zip(prow, packs)):
    ax.barh(i, r["pass_at_1"], color=col, height=0.6); ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color="#0b0b0b", lw=1); ax.text(r["ci_hi"] + 0.01, i, f"{r['pass_at_1']:.3f}", va="center", fontsize=7.5, color="#52514e")
ax.set_yticks(range(len(prow))); ax.set_yticklabels([f"{r['pack']} · {r['model']}" for r in prow]); ax.invert_yaxis(); ax.set_xlim(0, 1.08); ax.set_xlabel("pass@1 (60 paired groups; bars: bootstrap 95% CI)")
ax.set_title("Held-out confirmation packs (evaluation-only, 6,240 prompts each)", fontsize=9); fig.tight_layout(); fig.savefig(F/"fig4_packs.png"); fig.savefig(F/"fig4_packs.pdf"); plt.close(fig)

# Figure 5: regimes (no-thinking vs thinking) — 3 seeds per mixed-mode arm; thinking numbers from chain logs (26 / 52-prompt subsets)
MIX = {"mixed50": ["round11/report12_mixed50.json", "round13/report12_mixed50_s1.json", "round13/report12_mixed50_s2.json"],
       "distill": ["round12/report12_distill.json", "round13/report12_distill_s1.json", "round13/report12_distill_s2.json"],
       "distillsyn": ["round12/report12_distillsyn.json", "round13/report12_distillsyn_s1.json", "round13/report12_distillsyn_s2.json"]}
THINK = {  # name: (no-think mean, no-think sd, think26 mean, think26 sd, think60 mean, think60 sd, think tokens on think26, seeds)
    "base14b": (rep("round6/report12_base14b.json")["pass_at_1"], 0, 16/26, 0, 52/52, 0, 1757, 1),
    "random (v2)": (agg(R14["random"])[0], agg(R14["random"])[1], 18/26, 0, 23/52, 0, 999, 1),
    "mixed50": (*agg(MIX["mixed50"])[:2], 0.500, 0.000, 0.878, 0.056, 101, 3),
    "think100": (rep("round11/report12_think100.json")["pass_at_1"], 0, 17/26, 0, 18/52, 0, 82, 1),
    "distill": (*agg(MIX["distill"])[:2], 0.769, 0.038, 0.974, 0.022, 2005, 3),
    "distillsyn": (*agg(MIX["distillsyn"])[:2], 0.551, 0.022, 0.917, 0.059, 356, 3)}
with open(T/"table5_regimes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["adapter", "no_think_test_eval12", "sd", "think26_main", "sd", "think60_diag60", "sd", "think_tokens_think26", "seeds_thinking"])
    for k, v in THINK.items(): w.writerow([k, round(v[0], 3), round(v[1], 3), round(v[2], 3), round(v[3], 3), round(v[4], 3), round(v[5], 3), v[6], v[7]])
fig, ax = plt.subplots(figsize=(6.4, 3.0)); names = list(THINK); x = range(len(names)); wd = 0.26
cols = {"no-thinking (1,248)": C["random"], "thinking · main 26": C["mined"], "thinking · diag60 52": C["v3"]}
for j, (lab, col) in enumerate(cols.items()):
    ys = [THINK[n][2*j] for n in names]; es = [THINK[n][2*j+1] for n in names]; xs = [i + (j - 1) * wd for i in x]
    ax.bar(xs, ys, wd - .02, color=col, label=lab, yerr=es, capsize=2, error_kw=dict(lw=0.8, color="#52514e"))
    for xx, y, e in zip(xs, ys, es): ax.text(xx, y + e + .012, f"{y:.2f}", ha="center", fontsize=6.5, color="#52514e")
ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.set_ylim(0, 1.28); ax.set_ylabel("pass@1"); ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center")
ax.set_title("Qwen3-14B: no-thinking vs thinking regimes by target style (3 seeds for mixed-mode arms; sd bars)", fontsize=8.5)
fig.tight_layout(); fig.savefig(F/"fig5_regimes.png"); fig.savefig(F/"fig5_regimes.pdf"); plt.close(fig)

# Figure 6: v2 vs v3 seeds — boundary recognition vs overall accuracy at 14B
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for lab, paths, col, mk in (("v2 random", R14["random"], C["random"], "o"), ("v2 mined", R14["mined"], C["mined"], "o"), ("v3 random", V3["random"], C["random"], "^"), ("v3 mined", V3["mined"], C["mined"], "^")):
    xs = [rep(p)["per_view"]["boundary"] for p in paths]; ys = [rep(p)["pass_at_1"] for p in paths]
    ax.scatter(xs, ys, color=col, marker=mk, s=38, label=lab, edgecolor="white", lw=0.6)
ax.set_xlabel("boundary-view accuracy (status recognition)"); ax.set_ylabel("overall pass@1"); ax.set_xlim(0.7, 1.02); ax.set_ylim(0.28, 0.46)
ax.legend(frameon=False, fontsize=7.5); ax.set_title("Status-diverse (v3) training at 14B: boundary up, mean unchanged, variance up", fontsize=8.5)
fig.tight_layout(); fig.savefig(F/"fig6_v3_seeds.png"); fig.savefig(F/"fig6_v3_seeds.pdf"); plt.close(fig)

# Table 6: new-family transfer (dev112, native protocol lenient) typed from logs + pack-protocol lenient files
dev = [("Qwen3-8B base", 0.343, 0.326), ("8B sftv2_mined_s1", 0.220, 0.656), ("8B sftv3_random", None, 0.703), ("Qwen3-14B base", 0.575, None), ("14B sft14b_random", 0.491, None)]
with open(T/"table6_dev112_transfer.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "pack_protocol_2key_lenient", "native_protocol_4key_lenient"]); w.writerows(dev)
print("figures:", sorted(p.name for p in F.iterdir())); print("tables:", sorted(p.name for p in T.iterdir()))

# ---------- Energy-reasoning benchmark (de Zarza et al., Electronics 15(7):1395): published raw protocol (+ chat-template variant)
EA = json.load(open(D/"round15/energybench_analysis_all.json"))
EL = ["en", "es", "ca", "ar", "lb", "eu", "fr", "pt", "it", "de", "ru", "zh-cn", "zh-tw"]
PUB = {"Phi-4-mini (published)": .507, "Phi-3-mini (published)": .476, "Qwen2-7B (published)": .467, "Falcon-Mamba-7B (published)": .384, "Mistral-7B (published)": .332}
erows = []
for proto, od in (("raw", "out"), ("chat", "out_chat")):
    slots = sorted({k.split("/")[1] for k in EA if k.startswith(od + "/")})
    for s in slots:
        a = [EA[f"{od}/{s}/{l}"] for l in EL if f"{od}/{s}/{l}" in EA]
        if len(a) < 13: continue
        g = lambda key: st.mean(x[key] for x in a if isinstance(x.get(key), (int, float)))
        wh = st.mean(x["energy"]["wh_per_1000_queries"] for x in a if x.get("energy", {}).get("enabled"))
        erows.append(dict(protocol=proto, slot=s, mean=round(g("overall_average"), 3), step_accuracy=round(g("step_accuracy"), 3), consistency=round(g("consistency"), 4),
                          wh_per_1000_queries=round(wh, 1), **{l: round(x["overall_average"], 3) for l, x in zip(EL, a)}))
with open(T/"table7_energy_benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=erows[0].keys()); w.writeheader(); w.writerows(erows)
raw = {r["slot"]: r["mean"] for r in erows if r["protocol"] == "raw" and not r["slot"].startswith("phi-4-") or r["slot"] == "phi-4-mini-instruct" and r["protocol"] == "raw"}
order = [("Phi-4-mini (re-run)", raw.get("phi-4-mini-instruct"), C["ref"])] + [(k, v, "#c9c8c3") for k, v in PUB.items()] + \
        [("Qwen3-8B base", raw["qwen3-8b-base"], C["base"]), ("Qwen3-8B mined", raw["qwen3-8b-mined"], C["mined"]), ("Qwen3-8B v3 random", raw["qwen3-8b-v3random"], C["v3"]),
         ("Qwen3-14B base", raw["qwen3-14b-base"], C["base"]), ("Qwen3-14B random", raw["qwen3-14b-random"], C["random"]), ("Qwen3-14B mined", raw["qwen3-14b-mined"], C["mined"]),
         ("Qwen3-14B v3 mined", raw["qwen3-14b-v3mined"], C["v3"]), ("Qwen3-14B distill", raw["qwen3-14b-distill"], C["mixed"]), ("Qwen3-14B mixed50", raw["qwen3-14b-mixed50"], C["mixed"])]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for i, (n, v, col) in enumerate(order):
    ax.barh(i, v, color=col, height=0.65); ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=7.5, color="#52514e")
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[0] for o in order], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, .6)
ax.set_xlabel("overall score, mean of 13 languages (published raw-completion protocol)")
ax.set_title("Multilingual reasoning benchmark (Electronics 2026): derived vs published", fontsize=8.5)
fig.tight_layout(); fig.savefig(F/"fig7_energy_benchmark.png"); fig.savefig(F/"fig7_energy_benchmark.pdf"); plt.close(fig)
print("table7 rows", len(erows))

# Figure 8: chat-template variant of the energy-reasoning benchmark (paired base vs derived; Phi-4-mini as calibration)
chat = {r["slot"]: r["mean"] for r in erows if r["protocol"] == "chat"}
pairs = [("Phi-4", "phi-4-base", "phi-4-mined"), ("Qwen3-8B", "qwen3-8b-base", "qwen3-8b-mined"), ("Qwen3-14B", "qwen3-14b-base", "qwen3-14b-random")]
fig, ax = plt.subplots(figsize=(6.4, 2.8)); wd = 0.36
for i, (m, b, d) in enumerate(pairs):
    for j, (k, col, lab) in enumerate(((b, C["base"], "base"), (d, C["mined"], "derived (mined / random)"))):
        v = chat[k]; ax.bar(i + (j - .5) * wd, v, wd - .03, color=col, label=lab if i == 0 else None); ax.text(i + (j - .5) * wd, v + .008, f"{v:.3f}", ha="center", fontsize=7.5, color="#52514e")
ax.axhline(chat["phi-4-mini-instruct"], color=C["ref"], ls="--", lw=1); ax.text(2.55, chat["phi-4-mini-instruct"] + .008, f"Phi-4-mini {chat['phi-4-mini-instruct']:.3f}", ha="right", fontsize=7.5, color=C["ref"])
ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[0] for p in pairs]); ax.set_ylim(0, .6); ax.set_ylabel("overall score (13-language mean)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_title("Same benchmark, chat-template prompt (no thinking)", fontsize=9)
fig.tight_layout(); fig.savefig(F/"fig8_energy_chat.png"); fig.savefig(F/"fig8_energy_chat.pdf"); plt.close(fig)

# ---------- Standard multilingual benchmarks (lm-eval): Belebele (12 langs), Global-MMLU-Lite (8), MGSM native-CoT 8-shot (6, strict-match)
SB = list(csv.DictReader(open(D/"round15/stdbench_stdbench_summary.csv")))
LAB = {"qwen3-8b-base": "Qwen3-8B base", "qwen3-8b-mined": "Qwen3-8B mined", "qwen3-8b-v3random": "Qwen3-8B v3 random",
       "qwen3-14b-base": "Qwen3-14B base", "qwen3-14b-random": "Qwen3-14B random", "qwen3-14b-mined": "Qwen3-14B mined", "qwen3-14b-v3mined": "Qwen3-14B v3 mined",
       "qwen3-14b-distill": "Qwen3-14B distill", "qwen3-14b-mixed50": "Qwen3-14B mixed50", "phi-4-base": "Phi-4 base", "phi-4-mined": "Phi-4 mined",
       "phi-4-mini-instruct": "Phi-4-mini-instruct", "gemma-3-12b-it": "Gemma-3-12B-it", "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct", "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
       "aya-expanse-8b": "Aya-Expanse-8B", "phi-4-mini-mined": "Phi-4-mini mined", "phi-4-mini-random": "Phi-4-mini random", "qwen3-8b-v4": "Qwen3-8B v4", "phi-4-mini-v4": "Phi-4-mini v4", "qwen3-8b-v4b": "Qwen3-8B v4b", "phi-4-mini-v4b": "Phi-4-mini v4b", "eurollm-9b-instruct": "EuroLLM-9B-Instruct", "salamandra-7b-instruct": "Salamandra-7B-instruct"}
def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None
srows = [dict(model=LAB.get(r["model"], r["model"]), key=r["model"], belebele12=fnum(r["bele_mean"]), global_mmlu_lite8=fnum(r["gmmlu_mean"]),
              mgsm6_strict=fnum(r["mgsmstrict_mean"]), mgsm6_flexible=fnum(r["mgsm_mean"])) for r in SB]
with open(T/"table8_standard_benchmarks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "belebele12", "global_mmlu_lite8", "mgsm6_strict", "mgsm6_flexible"]); w.writeheader()
    for r in srows: w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "key"})
def scol(k):
    if k.startswith(("gemma", "qwen2.5", "llama", "aya", "eurollm", "salamandra")) or k == "phi-4-mini-instruct": return "#c9c8c3"
    if "v4" in k: return C["translated"]
    if k.endswith("base"): return C["base"]
    return C["mixed"] if ("mixed" in k or "distill" in k) else C["v3"] if "v3" in k else C["random"] if "random" in k else C["mined"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 5.0), sharey=True)
names = [r["model"] for r in srows]
for ax, (col, title, lo) in zip(axs, (("belebele12", "Belebele (12 langs)", .5), ("global_mmlu_lite8", "Global-MMLU-Lite (8)", .4), ("mgsm6_strict", "MGSM 8-shot (6), strict", .1))):
    for i, r in enumerate(srows):
        v = r[col]
        if v is None: continue
        ax.barh(i, v, color=scol(r["key"]), height=0.7); ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=6.5, color="#52514e")
    ax.set_xlim(lo, 1.0); ax.set_title(title, fontsize=8.5); ax.tick_params(axis="x", labelsize=7)
axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names, fontsize=7.5); axs[0].invert_yaxis()
fig.suptitle("Standard multilingual benchmarks: derived models (colour) vs bases (dark grey) and comparators (light grey)", fontsize=8.5)
fig.tight_layout(); fig.savefig(F/"fig9_standard_benchmarks.png"); fig.savefig(F/"fig9_standard_benchmarks.pdf"); plt.close(fig)


# ---------- Answer accuracy on the energy-reasoning benchmark (key value present) + base/mined/v4 comparison figure
AA = {(r["protocol"], r["slot"]): float(r["mean"]) for r in csv.DictReader(open(D/"round15/energybench_answer_accuracy.csv"))}
with open(T/"table9_answer_accuracy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["slot", "raw_answer_acc", "chat_answer_acc", "raw_semantic", "chat_semantic"])
    sem = {(r["protocol"], r["slot"]): r["mean"] for r in erows}
    for slot in sorted({k[1] for k in AA}):
        w.writerow([slot, AA.get(("out", slot)), AA.get(("out_chat", slot)), sem.get(("raw", slot)), sem.get(("chat", slot))])
SBK = {r["key"]: r for r in srows}
def semv(proto, slot): return next((r["mean"] for r in erows if r["protocol"] == proto and r["slot"] == slot), None)
groups = [("Phi-4-mini", ["phi-4-mini-instruct", "phi-4-mini-mined", "phi-4-mini-v4"] + (["phi-4-mini-v4b"] if ("out", "phi-4-mini-v4b") in AA else [])),
          ("Qwen3-8B", ["qwen3-8b-base", "qwen3-8b-mined", "qwen3-8b-v4"] + (["qwen3-8b-v4b"] if ("out", "qwen3-8b-v4b") in AA else []))]
metrics = [("semantic, raw", lambda k: semv("raw", k)), ("semantic, chat", lambda k: semv("chat", k)), ("answer acc., raw", lambda k: AA.get(("out", k))),
           ("answer acc., chat", lambda k: AA.get(("out_chat", k))), ("MGSM strict", lambda k: SBK[k]["mgsm6_strict"] if k in SBK else None)]
cols = ["#8a8985", C["mined"], C["translated"], C["random"]]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (gname, keys) in zip(axs, groups):
    wd = 0.8 / len(keys)
    for j, k in enumerate(keys):
        ys = [m(k) for _, m in metrics]
        ax.bar([i + (j - (len(keys) - 1) / 2) * wd for i in range(len(metrics))], [y or 0 for y in ys], wd * .92, color=cols[j], label=["base", "mined", "v4", "v4b"][j])
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m for m, _ in metrics], rotation=25, ha="right", fontsize=7); ax.set_title(gname, fontsize=9); ax.set_ylim(0, 1)
axs[0].set_ylabel("score"); axs[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
fig.suptitle("Format lock-in and its repair: base vs mined vs v4 vs v4b on external metrics", fontsize=8.5)
fig.tight_layout(); fig.savefig(F/"fig10_v4.png"); fig.savefig(F/"fig10_v4.pdf"); plt.close(fig)
