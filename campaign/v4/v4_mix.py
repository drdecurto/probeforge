"""ProbeForge v4 mixture: contrast JSON + contrast prose (format diversity) + verified multilingual GSM8K.

usage: python v4_mix.py --contrast CURRICULUM_DIR --gsm GSM_DIR --tokenizer MODEL --out OUT_DIR
        [--budget 3000000 --json-frac .5 --prose-frac .25 --gsm-frac .25 --seed 1]
Writes OUT_DIR/sft_train.jsonl, sft_validation.jsonl (contrast validation, unchanged) and mix_stats.json.
"""
import argparse, json, random, glob, os
from collections import Counter

PHRASE = {"en": "The answer is", "es": "La respuesta es", "ca": "La resposta és", "ar": "الإجابة هي",
          "lb": "D'Äntwert ass", "eu": "Erantzuna hau da:", "fr": "La réponse est", "pt": "A resposta é",
          "it": "La risposta è", "de": "Die Antwort lautet", "ru": "Ответ:", "zh-cn": "答案是", "zh-tw": "答案是"}
INSTR = {"en": 'Solve the problem step by step and finish with "The answer is" followed by the final number.',
         "es": 'Resuelve el problema paso a paso y termina con "La respuesta es" seguido del número final.',
         "ca": 'Resol el problema pas a pas i acaba amb "La resposta és" seguit del nombre final.',
         "ar": 'حل المسألة خطوة بخطوة واختم بعبارة "الإجابة هي" متبوعة بالعدد النهائي.',
         "lb": 'Léis de Problem Schrëtt fir Schrëtt a schléiss of mat "D\'Äntwert ass" gefollegt vun der Schlusszuel.',
         "eu": 'Ebatzi problema urratsez urrats eta amaitu "Erantzuna hau da:" esaldiarekin eta azken zenbakiarekin.',
         "fr": 'Résous le problème étape par étape et termine par "La réponse est" suivi du nombre final.',
         "pt": 'Resolve o problema passo a passo e termina com "A resposta é" seguido do número final.',
         "it": 'Risolvi il problema passo dopo passo e concludi con "La risposta è" seguito dal numero finale.',
         "de": 'Löse die Aufgabe Schritt für Schritt und schließe mit "Die Antwort lautet", gefolgt von der Endzahl.',
         "ru": 'Реши задачу пошагово и закончи фразой "Ответ:" с итоговым числом.',
         "zh-cn": "请逐步解答这道题，最后以“答案是”加上最终数字结尾。",
         "zh-tw": "請逐步解答這道題，最後以「答案是」加上最終數字結尾。"}


def to_prose(row):
    """Solved contrast row -> plain step-by-step prompt and prose completion; None if not convertible."""
    try:
        c = json.loads(row["completion"])
    except (json.JSONDecodeError, TypeError):
        return None
    if c.get("status") != "solved" or not isinstance(c.get("answer"), dict) or not c["answer"]:
        return None
    lang = row["language"]
    user = next(m["content"] for m in row["prompt"] if m["role"] == "user").split("\nJSON answer keys")[0].strip()
    ans = c["answer"]
    val = str(next(iter(ans.values()))) if len(ans) == 1 else ", ".join(f"{k} = {v}" for k, v in ans.items())
    end = "。" if lang.startswith("zh") else "."
    out = dict(row)
    out["prompt"] = [{"role": "system", "content": INSTR[lang]}, {"role": "user", "content": user}]
    out["completion"] = f"{c.get('explanation', '').strip()}\n{PHRASE[lang]} {val}{end}"
    out["id"] = row["id"] + "_prose"; out["kind"] = "prose"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", required=True); ap.add_argument("--gsm", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=3_000_000)
    ap.add_argument("--json-frac", type=float, default=.5); ap.add_argument("--prose-frac", type=float, default=.25)
    ap.add_argument("--gsm-frac", type=float, default=.25); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ntok = lambda r: len(tok.encode("\n".join(m["content"] for m in r["prompt"]) + "\n" + r["completion"]))
    rng = random.Random(a.seed)
    contrast = [json.loads(l) for l in open(f"{a.contrast}/sft_train.jsonl")]
    # unique source rows only (curricula repeat rows across exposures) -> prose copies come from distinct rows
    rng.shuffle(contrast)
    gsm = [json.loads(l) for f in sorted(glob.glob(f"{a.gsm}/gsm_*.jsonl")) for l in open(f)]
    for r in gsm:
        r.pop("gold", None)
        lines = r["completion"].rstrip().split("\n"); nz = [i for i, x in enumerate(lines) if x.strip()]
        if len(nz) >= 2 and lines[nz[-1]].strip() == lines[nz[-2]].strip():
            r["completion"] = "\n".join(lines[:nz[-1]]).rstrip()
    # balance GSM by language: round-robin over shuffled per-language lists
    by = {}
    for r in gsm: by.setdefault(r["language"], []).append(r)
    for v in by.values(): rng.shuffle(v)
    from itertools import zip_longest
    gsm_rr = [x for t in zip_longest(*by.values()) for x in t if x is not None]
    out, used, stats = [], set(), Counter()
    def take(rows, budget, tag, conv=None):
        t = 0
        for r in rows:
            if t >= budget: break
            if conv:
                if r["id"] in used: continue
                r = conv(r)
                if r is None: continue
            n = ntok(r); out.append(r); t += n; stats[f"{tag}_rows"] += 1; stats[f"{tag}_tokens"] += n
            if not conv and tag == "json": used.add(r["id"])
        return t
    take(contrast, a.budget * a.json_frac, "json")
    prose_src = [r for r in contrast if r["id"] not in used]
    take(prose_src, a.budget * a.prose_frac, "prose", to_prose)
    take(gsm_rr, a.budget * a.gsm_frac, "gsm")
    stats["gsm_languages"] = len(by); stats["gsm_available"] = len(gsm)
    rng.shuffle(out)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/sft_train.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.system(f"cp {a.contrast}/sft_validation.jsonl {a.out}/sft_validation.jsonl")
    json.dump(dict(stats), open(f"{a.out}/mix_stats.json", "w"), indent=1)
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
