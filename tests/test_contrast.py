import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from probeforge.io import canonical,read_jsonl,write_jsonl,write_json
from contrastforge.worlds import solve,make_world,variants,FAMILIES
from contrastforge.dataset import generate,gold,audit_rows,import_legacy
from contrastforge.verify import verify,parse
from contrastforge.collect import collect
from contrastforge.curriculum import prepare
from contrastforge.evaluate import evaluate,compare
from contrastforge.energy import integrate
from contrastforge.lic import generate_lic,utility,optimum,baseline_report


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def dataset(self):
        generate(self.root/"data",6,languages=["en","es","ca","ar","lb"])
        return [r for s in ("train","validation","test") for r in read_jsonl(self.root/f"data/{s}.jsonl")]

    def test_exact_answers_and_boundary_assumptions(self):
        def s(f,p):return solve({"family":f,"facts":p})
        self.assertEqual(s("product",{"total":6,"domain":"nonnegative"})[1],{"x":"4","y":"2","maximum":"32"})
        self.assertEqual(s("product",{"total":6,"domain":"real"})[0],"unbounded")
        self.assertEqual(s("affine",{"a":0,"b":1})[0],"infeasible")
        self.assertEqual(s("affine",{"a":0,"b":0})[0],"needs_clarification")
        self.assertEqual(s("variance",{"a":2,"b":4,"c":6,"convention":"population"})[1],{"variance":"8/3"})
        self.assertEqual(s("variance",{"a":2,"b":4,"c":6,"convention":"sample"})[1],{"variance":"4"})
        self.assertEqual(s("journey",{"distance_a":60,"distance_b":60,"speed_a":30,"speed_b":60})[1],{"speed":"40"})
        self.assertEqual(s("energy",{"capacity_wh":100,"watts":40,"efficiency":"4/5"})[1],{"hours":"2"})

    def test_product_maximum_against_dense_rational_grid(self):
        for total in (3,6,9,30):
            _,a=solve({"family":"product","facts":{"total":total,"domain":"nonnegative"}})
            best=F(a["maximum"])
            for j in range(total*20+1):
                x=F(j,20);self.assertLessEqual(x*x*(total-x),best)

    def test_family_splits_and_all_gold_responses(self):
        rows=self.dataset();self.assertTrue(audit_rows(rows)["passed"])
        for row in rows:self.assertTrue(verify(row,canonical(gold(row)))["correct"],row["id"])
        bad=copy.deepcopy(rows[0]);bad["id"]+="bad";bad["split"]="test"
        with self.assertRaises(ValueError):audit_rows(rows+[bad])

    def test_variants_change_semantics_and_unit_surface(self):
        for family in sum((list(x) for x in FAMILIES.values()),[]):
            for seed in range(8):
                v=variants(make_world(family,seed));base=solve(v[0][2])
                self.assertEqual(base,solve(v[1][2]));self.assertNotEqual(base,solve(v[3][2]));self.assertNotEqual(base,solve(v[4][2]))
        rows=self.dataset()
        for row in rows:
            if row["view"]=="units":
                base=next(r for r in rows if r["group_id"]==row["group_id"] and r["language"]==row["language"] and r["view"]=="base")
                self.assertNotEqual(row["prompt"],base["prompt"])
                self.assertEqual(gold(row),gold(base))

    def test_strict_numeric_and_schema_verifier(self):
        row=self.dataset()[0];obj=gold(row);key=next(iter(obj["answer"]))
        obj["answer"][key]="-"+obj["answer"][key]
        self.assertFalse(verify(row,canonical(obj))["correct"])
        self.assertFalse(verify(row,'{"status":"solved","status":"solved"}')["correct"])
        self.assertFalse(verify(row,'{"answer":NaN}')["correct"])
        self.assertFalse(verify(row,"The answer is somewhere in these numbers: 1 2 3 4")["correct"])
        with self.assertRaises(ValueError):parse('{"x":1,"x":2}')

    def test_collection_resume_mining_and_evaluation(self):
        self.dataset();inputs=[self.root/"data/train.jsonl",self.root/"data/validation.jsonl"]
        logs=self.root/"rollouts.jsonl"
        a=collect(inputs,logs,backend="mock");b=collect(inputs,logs,backend="mock")
        self.assertGreater(a["new_requests"],0);self.assertEqual(b["new_requests"],0)
        with self.assertRaises(ValueError):collect(inputs,logs,backend="mock",seed=99)
        with self.assertRaises(ValueError):prepare(inputs,self.root/"blocked",rollouts=[logs])
        result=prepare(inputs,self.root/"curr",rollouts=[logs],allow_synthetic=True)
        self.assertGreater(result["training_rows"],0);self.assertGreater(result["preference_counts"]["train"],0)
        r=evaluate(inputs,[logs],self.root/"eval.json",allow_synthetic=True)
        self.assertIsNone(r["budgets"]["512"]["total_joules"])
        self.assertLess(r["budgets"]["512"]["all_views_correct"],1)
        self.assertEqual(compare([self.root/"eval.json",self.root/"eval.json"],self.root/"comparison.json")["comparisons"][0]["mean_paired_group_difference"],0)

    def test_no_test_data_in_training(self):
        self.dataset()
        with self.assertRaises(ValueError):prepare([self.root/"data/train.jsonl",self.root/"data/test.jsonl"],self.root/"bad",strategy="random")

    def test_missing_and_truncated_count_as_failures(self):
        rows=self.dataset()[:3];path=self.root/"few.jsonl";write_jsonl(path,rows)
        logs=self.root/"logs.jsonl";collect([path],logs,backend="mock")
        records=list(read_jsonl(logs));records[0]["response"]=canonical(gold(rows[0]));records[0]["finish_reason"]="length"
        write_jsonl(logs,records[:-1])
        report=evaluate([path],[logs],self.root/"r.json",allow_synthetic=True)["budgets"]["512"]
        self.assertEqual(report["missing_requests"],1);self.assertEqual(report["length_stops"],1)
        self.assertLessEqual(report["pass_at_1"],1/3)

    def test_legacy_dedup_never_promotes_scores(self):
        p=self.root/"legacy.json"
        write_json(p,{"model":"m","problems":[{"problem":"q","solution":"wrong"}],"evaluations":[{"solutions":["a","a"],"average_score":1}]})
        report=import_legacy([p],self.root/"import.jsonl")
        self.assertEqual(report["distinct_responses"],1)
        self.assertFalse(next(read_jsonl(self.root/"import.jsonl"))["training_eligible"])

    def test_retention_import_is_exact_and_evaluation_only(self):
        from contrastforge.dataset import import_retention
        manifest=self.root/"retention.jsonl"
        write_jsonl(manifest,[{"source":"fixture@1","group_id":"a","task":"mc","language":"en",
                              "prompt":[{"role":"user","content":"Return JSON with answer A."}],"expected_response":{"answer":"A"}}])
        output=self.root/"retention_test.jsonl";import_retention(manifest,output)
        row=next(read_jsonl(output))
        self.assertTrue(verify(row,'{"answer":"A"}')["correct"])
        self.assertFalse(verify(row,'{"answer":"B"}')["correct"])
        with self.assertRaises(ValueError):prepare([output],self.root/"no_retention_train",strategy="random")

    def test_energy_integrates_time_not_problem_count(self):
        self.assertEqual(integrate([(0,100),(1,200),(3,100)]),450)
        self.assertIsNone(integrate([(0,100)]))
        with self.assertRaises(ValueError):integrate([(1,1),(0,1)])

    def test_api_retry_usage_and_remote_energy(self):
        import io,urllib.error
        from unittest.mock import patch
        from contrastforge.collect import APIBackend
        row=self.dataset()[0]
        response={"choices":[{"message":{"content":canonical(gold(row))},"finish_reason":"stop"}],
                  "usage":{"prompt_tokens":100,"completion_tokens":40},"model":"fixture"}
        failure=urllib.error.HTTPError("http://localhost/v1/chat/completions",429,"busy",{},None)
        with patch("urllib.request.urlopen",side_effect=[failure,io.BytesIO(json.dumps(response).encode())]),patch("time.sleep"):
            result=APIBackend("fixture","http://localhost/v1").generate(row,512,42,0)
        self.assertEqual(len(result["attempts"]),2)
        self.assertTrue(result["attempts"][0]["usage_unknown"])
        self.assertEqual(result["usage"]["completion_tokens"],40)
        self.assertIsNone(result["energy"]["joules"])
        self.assertTrue(verify(row,result["response"])["correct"])

    def test_lic_oracle_baselines_and_counterfactual(self):
        generate_lic(self.root/"lic",6,languages=["en"])
        rows=list(read_jsonl(self.root/"lic/train.jsonl"))
        for row in rows:self.assertTrue(verify(row,canonical(gold(row)))["correct"])
        records=baseline_report(rows)
        self.assertTrue(all(r.get("regret",0)>=0 for r in records))
        for changed in [r for r in rows if r["view"]=="counterfactual"]:
            base=next(r for r in rows if r["group_id"]==changed["group_id"] and r["view"]=="base")
            self.assertLess(utility(changed["state"],gold(base)),optimum(changed["state"])[1])

    def test_vision_grounding_reflection_and_scene_split(self):
        try:from PIL import Image
        except ImportError:self.skipTest("Optional Pillow not installed")
        from contrastforge.vision import prepare_vision,gold_vision,verify_vision
        imroot=self.root/"images";imroot.mkdir();manifest=[]
        for i,split in enumerate(("train","validation","test")):
            Image.new("RGB",(60,40),(i*50,0,0)).save(imroot/f"{i}.png")
            manifest.append({"image":f"{i}.png","scene_id":str(i),"split":split,"annotations_complete":True,
                             "vocabulary":["car","pedestrian"],"objects":[{"category":"car","bbox":[.1,.1,.2,.4]}]})
        path=self.root/"images.jsonl";write_jsonl(path,manifest)
        prepare_vision(path,imroot,self.root/"vlm",["en"])
        rows=list(read_jsonl(self.root/"vlm/train.jsonl"))
        base=next(r for r in rows if r["task"]=="side_counts" and r["view"]=="base")
        flip=next(r for r in rows if r["task"]=="side_counts" and r["view"]=="reflection")
        self.assertEqual(gold_vision(base)["counts"]["left"]["car"],1)
        self.assertEqual(gold_vision(flip)["counts"]["right"]["car"],1)
        self.assertFalse(verify_vision(base,gold_vision(flip))["correct"])
        self.assertFalse(verify_vision(base,{"counts":{"car":True}})["correct"])
        manifest[1]["scene_id"]="0";write_jsonl(path,manifest)
        with self.assertRaises(ValueError):prepare_vision(path,imroot,self.root/"bad_vlm",["en"])


if __name__=="__main__":unittest.main()
