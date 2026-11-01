import copy
import json
import random
from pathlib import Path

from .environment import messages, run_episode, parse_action, consistent
from .io import canonical, digest, read_jsonl, write_json, write_jsonl, file_hash
from .oracle import action_values
from .policies import RulePolicy, BudgetExceeded


def decision_row(task, decision, source):
    return {"prompt": messages(decision["state"]), "completion": decision["response"],
            "group_id": task["group_id"], "task_id": task["id"], "split": task["split"],
            "family": task["family"], "source": source}


def make_expert_data(tasks_path, out, penalty=.02):
    sft, prefs, episodes, seen = [], [], [], set()
    for task in read_jsonl(tasks_path):
        if task["split"] not in ("train", "validation"):
            raise ValueError("Training builders refuse test splits")
        episode = run_episode(task, RulePolicy("oracle", penalty))
        episodes.append({k: v for k, v in episode.items() if k != "decisions"})
        for dec in episode["decisions"]:
            if not dec["valid"]:
                continue
            row = decision_row(task, dec, "exact_planner")
            key = digest({"prompt": row["prompt"], "completion": row["completion"]})
            if key in seen:
                continue
            seen.add(key)
            sft.append(row)
            ranked = action_values(dec["state"], penalty)
            chosen = ranked[0]
            # At most two clear alternatives per prompt, no preference on ties.
            rejected = [r for r in ranked[1:] if chosen["value"] - r["value"] > .01]
            for r in rejected[-2:]:
                prefs.append({k: v for k, v in row.items() if k != "completion"} | {
                    "chosen": canonical(chosen["action"]), "rejected": canonical(r["action"]),
                    "margin": chosen["value"]-r["value"], "label_method": "exact_public_posterior_value"})
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft", sft), ("preferences", prefs), ("expert_episodes", episodes)):
        write_jsonl(root / f"{name}.jsonl", rows)
    summary = {"sft_rows": len(sft), "preference_rows": len(prefs), "episodes": len(episodes),
               "expert_success": sum(e["success"] for e in episodes)/max(1, len(episodes)),
               "source_sha256": file_hash(tasks_path), "cost_penalty": penalty}
    write_json(root / "manifest.json", summary)
    return summary


def collect(tasks_path, out, policy, limit=0, seed=0, resume=False, source="student"):
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "episodes.jsonl"
    identity = {"tasks_sha256": file_hash(tasks_path), "seed": seed, "source": source,
                "policy": getattr(policy, "model", None) if isinstance(getattr(policy, "model", None), str) else type(policy).__name__}
    # CLI stores complete generation arguments separately; identity protects inputs.
    if path.exists() and not resume:
        raise FileExistsError("Collection exists; pass --resume or use a fresh directory")
    prior = list(read_jsonl(path)) if path.exists() else []
    completed = {r["task_id"] for r in prior}
    if resume and (root / "identity.json").exists():
        if json.loads((root / "identity.json").read_text()) != identity:
            raise ValueError("Resume inputs differ")
    write_json(root / "identity.json", identity)
    count, stopped = 0, False
    with open(path, "a", encoding="utf-8") as stream:
        for task in read_jsonl(tasks_path):
            if task["split"] != "train":
                raise ValueError("Collection for training accepts train split only")
            if task["id"] in completed:
                continue
            if limit and count >= limit:
                break
            try:
                episode = run_episode(task, policy, seed=seed + int(digest(task["id"])[:8], 16))
            except BudgetExceeded:
                stopped = True
                break
            stream.write(canonical(episode) + "\n")
            stream.flush()
            count += 1
    # Verified successes only; original failures remain available for branching.
    rows = []
    for episode in read_jsonl(path):
        if episode["success"]:
            task_meta = {"id": episode["task_id"], **episode}
            rows.extend(decision_row(task_meta, d, source) for d in episode["decisions"] if d["valid"])
    write_jsonl(root / "sft_successes.jsonl", rows)
    report = {"new_episodes": count, "retained_decisions": len(rows), "budget_exhausted": stopped,
              "api_http_attempts": getattr(policy, "requests", None), "billed_usage": getattr(policy, "usage", None),
              "cache_hits": getattr(policy, "cache_hits", None)}
    write_json(root / "collection_report.json", report)
    return report


def branch_preferences(tasks_path, episodes_path, out, policy, max_states=100, worlds=4, repeats=2,
                       max_actions=4, penalty=.02, seed=42):
    """Paired interventions on a student's visited states, averaged over plausible worlds.

    Same posterior world and continuation seed for every alternative. Labels use
    final executable success; they are noisy estimates, not causal guarantees.
    """
    tasks = {t["id"]: t for t in read_jsonl(tasks_path)}
    if any(t["split"] != "train" for t in tasks.values()):
        raise ValueError("Branching accepts training tasks only")
    rng, rows, audits, seen = random.Random(seed), [], [], set()
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "preferences.jsonl").exists():
        raise FileExistsError("Use a fresh branch output directory")
    stopped = False
    for episode in read_jsonl(episodes_path):
        task = tasks[episode["task_id"]]
        for dec in episode["decisions"]:
            state = dec["state"]
            key = digest(state)
            if key in seen:
                continue
            if len(seen) >= max_states:
                break
            seen.add(key)
            ranked = action_values(state, penalty)
            stop = next(r["action"] for r in ranked if r["action"]["action"] == "answer")
            # Candidate pool includes student action, a public-posterior stop, and random probes.
            actions = [stop]
            try:
                student_action = parse_action(dec["response"])
                if student_action not in actions:
                    actions.append(student_action)
            except ValueError:
                pass
            pool = [r["action"] for r in ranked if r["action"]["action"] == "probe" and r["action"] not in actions]
            rng.shuffle(pool)
            actions = (actions + pool)[:max_actions]
            plausible = list(consistent(state))
            selected = rng.sample(plausible, min(worlds, len(plausible)))
            scored = []
            try:
                for action in actions:
                    samples = []
                    for wi in selected:
                        for rep in range(repeats):
                            world = {**task, "truth_index": wi}
                            result = run_episode(world, policy, start_state=state, first_action=action,
                                                 seed=seed + wi*1009 + rep*100003)
                            samples.append(float(result["success"]) - penalty*result["budget_spent"])
                    scored.append({"action": action, "value": sum(samples)/len(samples), "samples": samples})
            except BudgetExceeded:
                stopped = True
                break
            scored.sort(key=lambda r: -r["value"])
            audits.append({"state_hash": key, "task_id": task["id"], "world_indices": selected, "scored": scored})
            if len(scored) >= 2 and scored[0]["value"]-scored[-1]["value"] > .05:
                rows.append({"prompt": messages(state), "chosen": canonical(scored[0]["action"]),
                             "rejected": canonical(scored[-1]["action"]), "group_id": task["group_id"],
                             "split": "train", "family": task["family"], "source": "student_branching",
                             "label_method": "paired_student_continuations", "margin": scored[0]["value"]-scored[-1]["value"]})
            # Checkpoints after complete states survive an interrupted later state.
            write_jsonl(path / "preferences.jsonl", rows)
            write_jsonl(path / "audit.jsonl", audits)
        if stopped or len(seen) >= max_states:
            break
    write_jsonl(path / "preferences.jsonl", rows)
    write_jsonl(path / "audit.jsonl", audits)
    report = {"states": len(audits), "pairs": len(rows), "budget_exhausted": stopped,
              "worlds": worlds, "repeats": repeats, "cost_penalty": penalty,
              "api_http_attempts": getattr(policy, "requests", None), "billed_usage": getattr(policy, "usage", None)}
    write_json(path / "manifest.json", report)
    return report


def mix(inputs, out, seed=42, cap_per_source=0):
    rng, all_rows, seen = random.Random(seed), [], set()
    for path in inputs:
        rows = list(read_jsonl(path))
        if any(r["split"] != "train" for r in rows):
            raise ValueError("Mix accepts train rows only")
        rng.shuffle(rows)
        for row in rows[:cap_per_source or None]:
            key = digest({k: row.get(k) for k in ("prompt", "completion", "chosen", "rejected")})
            if key not in seen:
                seen.add(key)
                all_rows.append(row)
    rng.shuffle(all_rows)
    write_jsonl(out, all_rows)
    return {"rows": len(all_rows)}
