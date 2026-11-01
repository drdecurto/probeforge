"""Verified multilingual GSM8K (train split only) for ProbeForge v4.

usage: python gsm_ml.py --langs es,ca,ar --out DIR --model PATH [--n 3000]
For each language: translate (skip for en) -> keep only translations whose numbers are preserved ->
generate step-by-step solutions ending with the native answer phrase -> keep the first sample whose
final number equals the GSM8K gold answer. Writes DIR/gsm_<lang>.jsonl in contrastforge SFT format.
"""
import argparse, json, random, re, os
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
LNAME = {"es": "Spanish", "ca": "Catalan", "ar": "Arabic", "lb": "Luxembourgish", "eu": "Basque", "fr": "French",
         "pt": "European Portuguese", "it": "Italian", "de": "German", "ru": "Russian",
         "zh-cn": "Simplified Chinese", "zh-tw": "Traditional Chinese"}
CONCISE = "Be concise: at most 6 short steps, in plain text, without LaTeX or Markdown."
LATEX = ("$$", "\\frac", "\\(", "\\[", "\\times", "\\text", "\\boxed")
NUM = re.compile(r"\d[\d.,\s]*\d|\d")


def numbers(text):
    """Multiset of digit strings (separators removed) -- robust to 1,000 / 1.000 / 2,5 vs 2.5."""
    return Counter(re.sub(r"[.,\s]", "", m.group(0)) for m in NUM.finditer(text))


def final_number(text, lang):
    tail = text.split(PHRASE[lang])[-1] if PHRASE[lang] in text else text[-80:]
    m = re.findall(r"-?\d[\d.,]*", tail)
    if not m:
        return None
    v = m[0] if PHRASE[lang] in text else m[-1]
    v = v.rstrip(".,")
    # thousands separators: 1,000 / 1.000 / 1 000 -> 1000 ; decimal comma 2,5 -> 2.5
    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", v):
        v = re.sub(r"[.,]", "", v)
    v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return None


def dedupe_tail(t):
    """Drop a repeated final line ("La respuesta es 15.\n\nLa respuesta es 15.")."""
    lines = [x for x in t.rstrip().split("\n")]
    nz = [i for i, x in enumerate(lines) if x.strip()]
    if len(nz) >= 2 and lines[nz[-1]].strip() == lines[nz[-2]].strip():
        lines = lines[:nz[-1]]
    return "\n".join(lines).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-sol-tokens", type=int, default=1536)
    ap.add_argument("--concise", action="store_true", help="teacher-only addendum: short plain-text steps; reject LaTeX/long solutions")
    ap.add_argument("--max-kept-tokens", type=int, default=250)
    ap.add_argument("--dump-rejects", default="it,pt")
    a = ap.parse_args()
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    ds = load_dataset("openai/gsm8k", "main")["train"]
    idx = list(range(len(ds))); random.Random(20261022).shuffle(idx); idx = idx[:a.n]
    probs = [dict(gid=f"gsm8k_train_{i}", q=ds[i]["question"], gold=float(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))) for i in idx]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, seed=1234)
    tok = llm.get_tokenizer()
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    os.makedirs(a.out, exist_ok=True)
    for lang in a.langs.split(","):
        stats = Counter()
        if lang == "en":
            items = [dict(p, text=p["q"]) for p in probs]
        else:
            ps = [chat([{"role": "user", "content": f"Translate the following math word problem into {LNAME[lang]}. Keep every number exactly as written, in digits. Output only the translation.\n\n{p['q']}"}]) for p in probs]
            outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=700))
            items = []
            for p, o in zip(probs, outs):
                t = o.outputs[0].text.strip()
                if numbers(t) == numbers(p["q"]) and len(t) > 20:
                    items.append(dict(p, text=t)); stats["translated_ok"] += 1
                else:
                    stats["translation_rejected"] += 1
        sys_msg = INSTR[lang]
        teach = sys_msg + (" " + CONCISE if a.concise else "")
        ps = [chat([{"role": "system", "content": teach}, {"role": "user", "content": it["text"]}]) for it in items]
        rej = open(f"{a.out}/rejects_{lang}.jsonl", "w") if lang in a.dump_rejects.split(",") else None; nrej = 0
        outs = llm.generate(ps, SamplingParams(n=a.samples, temperature=0.7, top_p=0.95, max_tokens=a.max_sol_tokens, seed=7))
        with open(f"{a.out}/gsm_{lang}.jsonl", "w") as f:
            for it, o in zip(items, outs):
                keep = None
                for c in o.outputs:
                    txt = c.text.strip()
                    ok = c.finish_reason == "stop" and PHRASE[lang] in txt and final_number(txt, lang) is not None and abs(final_number(txt, lang) - it["gold"]) < 1e-6
                    if ok and a.concise and (len(c.token_ids) > a.max_kept_tokens or any(x in txt for x in LATEX)):
                        stats["rejected_style"] += 1; ok = False
                    if ok:
                        keep = txt; break
                if keep is None:
                    stats["no_verified_solution"] += 1
                    if rej is not None and nrej < 40:
                        c = o.outputs[0]; rej.write(json.dumps({"gold": it["gold"], "parsed": final_number(c.text, lang), "phrase_in": PHRASE[lang] in c.text,
                                                              "finish": c.finish_reason, "ntok": len(c.token_ids), "tail": c.text.strip()[-300:]}, ensure_ascii=False) + "\n"); nrej += 1
                    continue
                stats["kept"] += 1
                keep = dedupe_tail(keep)
                f.write(json.dumps({"id": f"{it['gid']}_{lang}", "group_id": it["gid"], "semantic_id": f"{it['gid']}:base", "language": lang,
                                    "family": "gsm8k", "kind": "gsm8k", "view": "base", "split": "train", "exposure_index": 0,
                                    "source_example_hash": it["gid"], "prompt": [{"role": "system", "content": sys_msg}, {"role": "user", "content": it["text"]}],
                                    "completion": keep, "gold": it["gold"]}, ensure_ascii=False) + "\n")
        print(lang, dict(stats), flush=True)


if __name__ == "__main__":
    main()
