import json
from pathlib import Path
from probeforge.io import digest, read_jsonl, write_json, write_jsonl, canonical
from .worlds import FAMILIES, make_world, variants, solve, rational
from .languages import LANGUAGES, INTRO, EXPLANATIONS, index, render

ANSWER_KEYS = {"rate": ["speed_kmh"], "affine": ["x"], "mean": ["mean"],
 "variance": ["variance"], "product": ["x", "y", "maximum"], "energy": ["hours"],
 "workrate": ["hours"], "rectangle": ["side", "maximum_area"], "system": ["x", "y"],
 "journey": ["speed"], "production": ["profit", "x", "y"], "interest": ["balance"]}


def normalized_facts(world):
    return {k: v if v is None or k in ("domain", "convention") else rational(v)
            for k, v in world["facts"].items()}


def gold(row):
    if row["kind"] == "retention": return row["expected_response"]
    if row["kind"] == "lic":
        from .lic import gold_policy
        return gold_policy(row)
    if row["kind"] == "vlm":
        from .vision import gold_vision
        return gold_vision(row)
    status, answer = solve(row["world"])
    return {"status": status, "answer": answer, "facts": normalized_facts(row["world"]),
            "explanation": EXPLANATIONS[status][index(row["language"])]}


def generate(out, groups=1000, seed=42, languages=None, metamorphic=False, status_diverse=False):
    """groups is the number of training base worlds; validation/test each use groups//5."""
    root = Path(out)
    if root.exists() and any(root.iterdir()): raise FileExistsError("Dataset directory is nonempty")
    languages = languages or list(LANGUAGES)
    if len(set(languages)) != len(languages) or not set(languages) <= set(LANGUAGES):
        raise ValueError("Languages must be distinct supported codes")
    if groups < 6: raise ValueError("Use at least six training groups")
    manifest = {"schema_version": 2, "seed": seed, "languages": languages,
                "split_policy": "disjoint problem families; all views remain with their root",
                "translation_review": "authored templates; bilingual review required", "metamorphic_views": bool(metamorphic), "status_diverse_views": bool(status_diverse), "splits": {}}
    all_rows = []
    for split, families in FAMILIES.items():
        n = groups if split == "train" else max(6, groups//5)
        rows, seen = [], set()
        for j in range(n):
            family = families[j % len(families)]
            attempt = 0
            while True:
                world = make_world(family, int(digest([seed, split, j, attempt])[:12], 16))
                group = digest(world)[:24]
                if group not in seen: break
                attempt += 1
                if attempt > 10000: raise ValueError("Requested more unique worlds than the generator supports")
            seen.add(group)
            for view, semantic, variant in variants(world, metamorphic, status_diverse):
                for lang in languages:
                    keys = ", ".join(ANSWER_KEYS[family])
                    fields = ", ".join(variant["facts"])
                    prompt = [{"role": "system", "content": INTRO[lang]},
                              {"role": "user", "content": render(variant, lang, view)+
                               f"\nJSON answer keys: {keys}. JSON facts keys: {fields}."+
                               "\nCanonical facts units: km, h, Wh, W. Missing facts: null."}]
                    row = {"id": digest([group, view, lang])[:24], "group_id": group,
                           "semantic_id": group+":"+semantic, "split": split, "family": family,
                           "language": lang, "view": view, "kind": "math", "world": variant,
                           "prompt": prompt, "source": "authored_generator_v2"}
                    rows.append(row)
        write_jsonl(root/f"{split}.jsonl", rows)
        manifest["splits"][split] = {"rows": len(rows), "groups": n, "families": list(families)}
        all_rows.extend(rows)
    audit_rows(all_rows)
    manifest["dataset_hash"] = digest(all_rows)
    write_json(root/"manifest.json", manifest)
    return manifest


def audit_rows(rows):
    if not rows: raise ValueError("Empty dataset")
    ids, groups, families = set(), {}, {}
    for r in rows:
        if r["id"] in ids: raise ValueError("Duplicate example ID")
        ids.add(r["id"])
        checks = [("group_id", groups)]
        if r["kind"] != "vlm": checks.append(("family", families))
        for key, container in checks:
            old = container.setdefault(r[key], r["split"])
            if old != r["split"]: raise ValueError(f"{key} leakage: {r[key]}")
        if r["split"] not in ("train", "validation", "test", "diagnostic"):
            raise ValueError("Invalid split")
        if r["language"] not in LANGUAGES: raise ValueError("Unsupported language")
    return {"rows": len(rows), "groups": len(groups), "families": len(families), "passed": True}


def sft_row(row, completion=None):
    return {k: row[k] for k in ("id", "group_id", "semantic_id", "split", "family", "language", "view", "kind", "prompt")} | {
        "completion": completion or canonical(gold(row)), "source_example_hash": digest(row)}


def import_legacy(paths, out):
    """Extract and deduplicate old responses for diagnostics ONLY; scores are not labels."""
    result, seen, problems, missing = [], set(), set(), 0
    for path in paths:
        root = Path(path)
        for file in sorted(root.rglob("*.json")) if root.is_dir() else [root]:
            try: data = json.loads(file.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError): continue
            if not isinstance(data, dict): continue
            pp, ee = data.get("problems", []), data.get("evaluations", [])
            if len(pp) != len(ee):
                if ee: raise ValueError(f"Cannot align problems and evaluations: {file}")
                continue
            for p, e in zip(pp, ee):
                question = p.get("problem", "")
                problems.add(question)
                if e is None: missing += 1; continue
                for text in e.get("solutions", []):
                    if not isinstance(text, str): continue
                    key = digest([question, text])
                    if key in seen: continue
                    seen.add(key)
                    result.append({"id": key, "split": "diagnostic", "question": question,
                                   "response": text, "reference_unverified": p.get("solution"),
                                   "model": data.get("model"), "source": str(file),
                                   "training_eligible": False})
    write_jsonl(out, result)
    return {"distinct_problems": len(problems), "distinct_responses": len(result), "missing_records": missing}


def import_retention(path,out):
    """Reviewed exact-JSON external tasks; permanently evaluation-only."""
    rows=[]
    for item in read_jsonl(path):
        if not isinstance(item["expected_response"],dict):raise ValueError("Retention target must be a JSON object")
        if item["language"] not in LANGUAGES or not item["prompt"]:raise ValueError("Invalid retention prompt/language")
        group=digest(["retention",item["source"],item["group_id"]])[:24]
        rows.append({"id":digest(item)[:24],"group_id":group,"semantic_id":group,
                     "kind":"retention","split":"test","family":"retention_"+item["task"],
                     "language":item["language"],"view":"base","prompt":item["prompt"],
                     "expected_response":item["expected_response"],"source":item["source"]})
    audit_rows(rows);write_jsonl(out,rows)
    return {"rows":len(rows),"split":"test","grader":"reviewed exact JSON; not a general natural-language judge"}
