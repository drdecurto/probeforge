"""Regenerate external plots (7--9) from included round 14 summaries."""
import json,csv,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
D=ROOT/"evidence"; F=ROOT/"figures"; T=ROOT/"source_tables"
C={"base":"#8a8985","mined":"#eb6834","random":"#2a78d6","v3":"#eda100","mixed":"#e87ba4","ref":"#52514e"}
# ---------- Energy-reasoning benchmark (de Zarza et al., Electronics 15(7):1395): published raw protocol (+ chat-template variant)
EA = json.load(open(D/"round14/analysis_all.json"))
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
SB = list(csv.DictReader(open(D/"round14/stdbench_summary.csv")))
LAB = {"qwen3-8b-base": "Qwen3-8B base", "qwen3-8b-mined": "Qwen3-8B mined", "qwen3-8b-v3random": "Qwen3-8B v3 random",
       "qwen3-14b-base": "Qwen3-14B base", "qwen3-14b-random": "Qwen3-14B random", "qwen3-14b-mined": "Qwen3-14B mined", "qwen3-14b-v3mined": "Qwen3-14B v3 mined",
       "qwen3-14b-distill": "Qwen3-14B distill", "qwen3-14b-mixed50": "Qwen3-14B mixed50", "phi-4-base": "Phi-4 base", "phi-4-mined": "Phi-4 mined",
       "phi-4-mini-instruct": "Phi-4-mini-instruct", "gemma-3-12b-it": "Gemma-3-12B-it", "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct", "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
       "aya-expanse-8b": "Aya-Expanse-8B", "eurollm-9b-instruct": "EuroLLM-9B-Instruct", "salamandra-7b-instruct": "Salamandra-7B-instruct"}
def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None
srows = [dict(model=LAB.get(r["model"], r["model"]), key=r["model"], belebele12=fnum(r["bele_mean"]), global_mmlu_lite8=fnum(r["gmmlu_mean"]),
              mgsm6_strict=fnum(r["mgsmstrict_mean"]), mgsm6_flexible=fnum(r["mgsm_mean"])) for r in SB]
with open(T/"table8_standard_benchmarks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "belebele12", "global_mmlu_lite8", "mgsm6_strict", "mgsm6_flexible"]); w.writeheader()
    for r in srows: w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "key"})
def scol(k):
    if k.startswith(("gemma", "qwen2.5", "llama", "aya", "eurollm", "salamandra", "phi-4-mini")): return "#c9c8c3"
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
