"""Strict semantic verification; explanations never establish correctness."""
import json
from .worlds import number, solve
from .dataset import normalized_facts


def parse(text):
    if not isinstance(text, str) or len(text) > 100000: raise ValueError("Invalid response size")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    def unique(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError("Duplicate JSON key")
            out[k] = v
        return out
    out = json.loads(text, object_pairs_hook=unique,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite number")))
    if not isinstance(out, dict): raise ValueError("Expected JSON object")
    return out


def equal_facts(got, expected):
    if not isinstance(got, dict) or set(got) != set(expected): return False
    for k, value in expected.items():
        if value is None:
            if got[k] is not None: return False
        elif k in ("domain", "convention"):
            if got[k] != value: return False
        else:
            try:
                if number(got[k]) != number(value): return False
            except (ValueError, ZeroDivisionError, TypeError): return False
    return True


def verify(row, text):
    try:
        obj = parse(text)
        if row["kind"] == "retention":
            from probeforge.io import canonical
            correct=canonical(obj)==canonical(row["expected_response"])
            return {"correct":correct,"reason":"verified_retention" if correct else "retention_mismatch"}
        if row["kind"] == "lic":
            from .lic import verify_policy
            return verify_policy(row, obj)
        if row["kind"] == "vlm":
            from .vision import verify_vision
            return verify_vision(row, obj)
        if set(obj) != {"status", "answer", "facts", "explanation"}:
            return {"correct": False, "reason": "schema"}
        if not isinstance(obj["explanation"], str): return {"correct": False, "reason": "schema"}
        status, answer = solve(row["world"])
        facts_ok = equal_facts(obj["facts"], normalized_facts(row["world"]))
        status_ok = obj["status"] == status
        answer_ok = equal_facts(obj["answer"], answer)
        correct = facts_ok and status_ok and answer_ok
        return {"correct": correct, "reason": "verified" if correct else "semantic_mismatch",
                "facts_correct": facts_ok, "status_correct": status_ok, "answer_correct": answer_ok,
                "expected_status": status, "explanation_verified": False}
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError, RecursionError):
        return {"correct": False, "reason": "invalid_response"}
