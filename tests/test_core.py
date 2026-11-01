import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from probeforge.tasks import generate_dataset, evaluate_rule
from probeforge.environment import public_state, run_episode, messages, parse_action, predictions, consistent
from probeforge.policies import RulePolicy, APIPolicy, BudgetExceeded
from probeforge.oracle import action_values
from probeforge.validation import audit_dataset
from probeforge.io import canonical, read_jsonl
from probeforge.data import make_expert_data, collect, branch_preferences
from probeforge.evaluation import evaluate, compare


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generate_dataset(self.root/"tasks", 36, 12)
        self.tasks = list(read_jsonl(self.root/"tasks/train.jsonl"))

    def tearDown(self):
        self.temp.cleanup()

    def test_split_audit_and_reproducibility(self):
        self.assertTrue(audit_dataset(self.root/"tasks")["passed"])
        generate_dataset(self.root/"again", 36, 12)
        self.assertEqual((self.root/"tasks/train.jsonl").read_bytes(), (self.root/"again/train.jsonl").read_bytes())

    def test_counterfactual_worlds_have_identical_prompts(self):
        a, b = self.tasks[:2]
        self.assertEqual(public_state(a), public_state(b))
        self.assertNotIn("truth_index", canonical(messages(public_state(a))))
        self.assertNotIn(a["id"], canonical(messages(public_state(a))))

    def test_truth_does_not_change_expert_initial_action(self):
        outputs = [RulePolicy().generate(messages(public_state(t)))[0] for t in self.tasks[:8]]
        self.assertEqual(len(set(outputs)), 1)

    def test_oracle_dominates_no_information_on_complete_groups(self):
        oracle = [run_episode(t, RulePolicy(), budget=4) for t in self.tasks]
        direct = [run_episode(t, RulePolicy("no_probe"), budget=4) for t in self.tasks]
        self.assertGreater(sum(r["success"] for r in oracle), sum(r["success"] for r in direct))
        self.assertTrue(all(r["budget_spent"] <= 4 for r in oracle))
        self.assertTrue(all(r["invalid_actions"] == 0 for r in oracle))

    def test_invalid_probe_is_failure(self):
        class Bad:
            def generate(self, *a, **k):
                return '{"action":"probe","probe_id":"does-not-exist"}', {}
        result = run_episode(self.tasks[0], Bad())
        self.assertFalse(result["success"])
        self.assertEqual(result["invalid_actions"], 1)
        self.assertEqual(result["tool_calls"], 0)

    def test_budget_zero_stops(self):
        result = run_episode(self.tasks[0], RulePolicy(), budget=0)
        self.assertEqual(result["tool_calls"], 0)

    def test_random_probe_intervention_preserves_cost(self):
        from probeforge.policies import RandomProbeIntervention
        state = public_state(self.tasks[0], remaining=4)
        original, _ = RulePolicy().generate(messages(state))
        changed, usage = RandomProbeIntervention(RulePolicy()).generate(messages(state), seed=19)
        a, b = parse_action(original), parse_action(changed)
        if a["action"] == "probe":
            costs = {p["id"]: p["cost"] for p in state["probes"]}
            self.assertEqual(costs[a["probe_id"]], costs[b["probe_id"]])
            self.assertEqual(usage["policy_proposal"], original)

    def test_training_split_validation_rejects_overlap(self):
        from probeforge.train import validate_splits
        with self.assertRaises(ValueError):
            validate_splits([{"split":"train","group_id":"same"}], [{"split":"validation","group_id":"same"}])

    def test_parser_rejects_extra_prose_and_bool(self):
        for text in ['Sure {"action":"answer"}', '{"action":"answer","predictions":[true,1,2]}', '{"action":"probe","probe_id":"p0","observation":4}']:
            with self.assertRaises(ValueError):
                parse_action(text)

    def test_rule_arithmetic(self):
        rule = {"family": "battery_efficiency", "parameters": [3, 2, 1]}
        self.assertEqual(evaluate_rule(rule, [10, 3, 1]), 11)
        rule = {"family": "sequence_weighted", "parameters": [1, 2, 3]}
        self.assertEqual(evaluate_rule(rule, [1, 1]), 11)

    def test_build_collection_branch_and_evaluation(self):
        summary = make_expert_data(self.root/"tasks/train.jsonl", self.root/"expert")
        self.assertGreater(summary["sft_rows"], 0)
        prefs = list(read_jsonl(self.root/"expert/preferences.jsonl"))
        self.assertTrue(all(p["margin"] > .01 for p in prefs))
        report = collect(self.root/"tasks/train.jsonl", self.root/"collect", RulePolicy(), limit=8)
        self.assertGreater(report["retained_decisions"], 0)
        branch = branch_preferences(self.root/"tasks/train.jsonl", self.root/"collect/episodes.jsonl", self.root/"branch", RulePolicy(), max_states=3, worlds=2, repeats=1)
        self.assertEqual(branch["states"], 3)
        evaluate(self.root/"tasks/test_ood.jsonl", self.root/"eval-a", RulePolicy("no_probe"), budgets=[0, 2], max_groups=2)
        evaluate(self.root/"tasks/test_ood.jsonl", self.root/"eval-b", RulePolicy(), budgets=[0, 2], max_groups=2)
        report = compare(self.root/"eval-a/episodes.jsonl", self.root/"eval-b/episodes.jsonl", self.root/"delta.json")
        self.assertEqual(report[0]["B_minus_A"], 0)
        self.assertGreaterEqual(report[1]["B_minus_A"], 0)

    def test_api_contract_cache_and_budget(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(payload)
                response = {"choices": [{"message": {"content": '{"action":"answer","predictions":[1,2,3]}'}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 8}}
                self.send_response(200); self.end_headers(); self.wfile.write(canonical(response).encode())
            def log_message(self, *a):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            p = APIPolicy("test-model", f"http://127.0.0.1:{server.server_port}/v1", cache=self.root/"cache", max_requests=1)
            text, usage = p.generate(messages(public_state(self.tasks[0])))
            self.assertEqual(usage["completion_tokens"], 8)
            self.assertEqual(p.generate(messages(public_state(self.tasks[0])))[0], text)
            self.assertEqual(p.requests, 1)
            self.assertEqual(p.cache_hits, 1)
            with self.assertRaises(BudgetExceeded):
                p.generate(messages(public_state(self.tasks[0])), seed=1)
            self.assertNotIn("truth_index", canonical(received))
        finally:
            server.shutdown(); worker.join(); server.server_close()


if __name__ == "__main__":
    unittest.main()
