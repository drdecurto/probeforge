import copy
import json
from .tasks import evaluate_rule
from .io import canonical

SYSTEM = '''You investigate a deterministic hidden system. Exactly one listed candidate rule is true.
All candidates have equal prior probability. Use actual observations to predict the three target outputs.
The formula uses integer arithmetic: // is floor division; % is Python nonnegative modulo;
int(condition) is 1 if true and 0 otherwise; sequence indices start at zero.
Battery input is [s, charge, demand] in normalized integer units, not physical units.
You may query only listed probes, paying their costs. Select useful experiments within the remaining budget.
Stop when sufficient evidence is available. If ambiguous, make your best prediction.
Reply with exactly one JSON object and no other text:
{"action":"probe","probe_id":"p0"} OR {"action":"answer","predictions":[integer,integer,integer]}.
Do not invent tool observations. The runtime will execute probes and return their results.'''


def public_state(task, observations=None, remaining=None):
    return {"family": task["family"], "formula": task["formula"],
            "candidate_parameters": [r["parameters"] for r in task["candidates"]],
            "parameter_order": ["a", "b", "c"], "probes": task["probes"],
            "targets": task["targets"], "observations": copy.deepcopy(task["initial"] if observations is None else observations),
            "remaining_budget": task["budget"] if remaining is None else remaining}


def rules(state):
    return [{"family": state["family"], "parameters": p} for p in state["candidate_parameters"]]


def consistent(state):
    return tuple(i for i, r in enumerate(rules(state)) if all(evaluate_rule(r, o["input"]) == o["output"] for o in state["observations"]))


def predictions(state, index):
    rule = rules(state)[index]
    return [evaluate_rule(rule, x) for x in state["targets"]]


def available_probes(state):
    used = {canonical(o["input"]) for o in state["observations"]}
    return [p for p in state["probes"] if p["cost"] <= state["remaining_budget"] and canonical(p["input"]) not in used]


def observe(state, probe, value):
    result = copy.deepcopy(state)
    result["observations"].append({"input": probe["input"], "output": value, "probe_id": probe["id"]})
    result["remaining_budget"] -= probe["cost"]
    return result


def messages(state):
    # A full sufficient state is resent each turn; no hidden evaluator keys appear.
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": canonical(state)}]


def parse_action(text):
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError("Response must be a short JSON string")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Action must be an object")
    if value.get("action") == "probe" and set(value) == {"action", "probe_id"} and isinstance(value["probe_id"], str):
        return value
    if value.get("action") == "answer" and set(value) == {"action", "predictions"}:
        p = value["predictions"]
        if isinstance(p, list) and len(p) == 3 and all(type(v) is int and abs(v) < 10**9 for v in p):
            return value
    raise ValueError("Invalid action schema")


def run_episode(task, policy, budget=None, start_state=None, first_action=None, seed=0, max_turns=8):
    state = copy.deepcopy(start_state) if start_state is not None else public_state(task, remaining=budget)
    initial_budget = state["remaining_budget"]
    target = predictions(state, task["truth_index"])
    decisions, calls, invalid, generated, prompt_tokens = [], 0, 0, 0, 0
    token_usage_available = True
    for turn in range(max_turns):
        before = copy.deepcopy(state)
        usage = {}
        if turn == 0 and first_action is not None:
            text = canonical(first_action)
        else:
            text, usage = policy.generate(messages(state), seed=seed + turn)
            generated += usage.get("completion_tokens", 0)
            prompt_tokens += usage.get("prompt_tokens", 0)
            token_usage_available = token_usage_available and usage.get("token_usage_available", all(k in usage for k in ("prompt_tokens", "completion_tokens")))
        try:
            action = parse_action(text)
            if action["action"] == "probe":
                probe = next((p for p in available_probes(state) if p["id"] == action["probe_id"]), None)
                if probe is None:
                    raise ValueError("Probe unavailable, repeated, or unaffordable")
                value = evaluate_rule(task["candidates"][task["truth_index"]], probe["input"])
                state = observe(state, probe, value)
                calls += 1
                decisions.append({"state": before, "response": text, "valid": True, "observation": value,
                                  **({"policy_proposal": usage["policy_proposal"]} if "policy_proposal" in usage else {})})
            else:
                decisions.append({"state": before, "response": text, "valid": True})
                return {"task_id": task["id"], "group_id": task["group_id"], "family": task["family"],
                        "split": task["split"], "success": action["predictions"] == target,
                        "prediction": action["predictions"], "target": target, "decisions": decisions,
                        "tool_calls": calls, "budget_spent": initial_budget-state["remaining_budget"],
                        "invalid_actions": invalid, "completion_tokens": generated, "prompt_tokens": prompt_tokens,
                        "token_usage_available": token_usage_available}
        except ValueError as exc:
            invalid += 1
            decisions.append({"state": before, "response": text, "valid": False, "error": str(exc)})
            # Invalid output is a failed episode, not a hidden unlimited retry.
            break
    return {"task_id": task["id"], "group_id": task["group_id"], "family": task["family"], "split": task["split"],
            "success": False, "decisions": decisions, "tool_calls": calls,
            "budget_spent": initial_budget-state["remaining_budget"], "invalid_actions": invalid,
            "completion_tokens": generated, "prompt_tokens": prompt_tokens, "exhausted": True,
            "token_usage_available": token_usage_available}
