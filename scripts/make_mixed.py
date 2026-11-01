"""Mixed-mode SFT targets: render a fraction of curriculum rows under the reasoning template with a <think> block,
followed by the usual derivation-first worked JSON. The think block is either
  * synthetic: the language-neutral derivation trace (derive.derivation), or
  * distilled: the base model's own verified-correct reasoning trace (collect --thinking rollouts, `thinking` field),
    optionally with the synthetic derivation appended (--append-derivation).
Evaluation is unchanged (collect strips the think block; the verifier ignores it).

usage: make_mixed.py CURRICULUM_DIR DATASET_DIR OUT_DIR [--frac 0.5] [--seed 1]
                     [--traces ROLLOUTS.jsonl ...] [--fill synthetic|none] [--append-derivation] [--max-trace-tokens N]
With --traces: rows whose id has a correct, finished trace get it (up to --frac of rows); --fill synthetic fills the
remaining quota with synthetic traces, --fill none leaves them plain.
"""
import argparse, json, random
from pathlib import Path
from contrastforge.derive import derivation

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("curriculum"); ap.add_argument("dataset"); ap.add_argument("out")
    ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--traces", nargs="*", default=[]); ap.add_argument("--fill", choices=("synthetic", "none"), default="synthetic")
    ap.add_argument("--append-derivation", action="store_true"); ap.add_argument("--max-trace-chars", type=int, default=6000)
    a = ap.parse_args()
    worlds = {}
    for part in ("train", "validation"):
        for line in open(Path(a.dataset)/f"{part}.jsonl"):
            r = json.loads(line); worlds[r["id"]] = r["world"]
    traces = {}
    for path in a.traces:
        for line in open(path):
            rec = json.loads(line); t = rec.get("thinking")
            if t and rec.get("finish_reason") == "stop" and rec.get("verification", {}).get("correct") and len(t) <= a.max_trace_chars:
                traces.setdefault(rec["id"], t)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed); stats = {"traces_available": len(traces)}
    for part in ("train", "validation"):
        rows = [json.loads(l) for l in open(Path(a.curriculum)/f"sft_{part}.jsonl")]
        quota = int(round(a.frac*len(rows))); nd = ns = 0
        order = list(range(len(rows))); rng.shuffle(order)
        with_trace = [i for i in order if rows[i]["id"] in traces]; without = [i for i in order if rows[i]["id"] not in traces]
        for i in with_trace[:quota]:
            r = rows[i]; think = traces[r["id"]]
            if a.append_derivation: think = think.rstrip() + "\n\nSo: " + derivation(worlds[r["id"]])
            r["thinking"] = True; r["think_source"] = "distilled"; r["completion"] = f"<think>\n{think}\n</think>\n\n{r['completion']}"; nd += 1
        if a.fill == "synthetic":
            for i in without[:max(0, quota-nd)]:
                r = rows[i]
                if r.get("kind", "math") != "math": continue
                r["thinking"] = True; r["think_source"] = "synthetic"; r["completion"] = f"<think>\n{derivation(worlds[r['id']])}\n</think>\n\n{r['completion']}"; ns += 1
        with open(out/f"sft_{part}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n")
        stats[part] = {"rows": len(rows), "distilled_rows": nd, "synthetic_rows": ns}
    m = json.load(open(Path(a.curriculum)/"manifest.json")) if (Path(a.curriculum)/"manifest.json").exists() else {}
    m.update(mixed_mode={"frac": a.frac, "seed": a.seed, "traces": a.traces, "fill": a.fill, "append_derivation": a.append_derivation, **stats})
    json.dump(m, open(out/"manifest.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(stats))

if __name__ == "__main__": main()
