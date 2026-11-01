"""Lenient re-scoring for the coverage-extension pack (kind=retention):
strict = stock verifier (exact object equality); lenient = ignore extra keys 'facts'/'explanation'
and compare only status + answer (numbers compared as exact rationals). Usage:
  python rescore_lenient.py <rows.jsonl> <rollouts.jsonl> [--out report.json]"""
import json, sys, collections
from fractions import Fraction
def parse(t):
    t=t.strip()
    if t.startswith("```json\n") and t.endswith("\n```"): t=t[8:-4]
    return json.loads(t)
def num_eq(a,b):
    try: return Fraction(str(a))==Fraction(str(b))
    except Exception: return a==b
def lenient(obj,exp):
    if not isinstance(obj,dict) or obj.get("status")!=exp["status"]: return False
    a,e=obj.get("answer"),exp["answer"]
    if not isinstance(a,dict) or set(a)!=set(e): return False
    return all(num_eq(a[k],e[k]) for k in e)
rows={}
for l in open(sys.argv[1]): r=json.loads(l); rows[r["id"]]=r
recs=[json.loads(l) for l in open(sys.argv[2])]
tot=collections.Counter(); strict=collections.Counter(); len_=collections.Counter(); fmt=collections.Counter()
by=lambda r,k: rows[r["id"]][k]
for r in recs:
    row=rows[r["id"]]; exp=row["expected_response"]
    try: obj=parse(r["response"]); ok_l=lenient(obj,exp); parsed=True
    except Exception: obj=None; ok_l=False; parsed=False
    ok_s=bool(r["verification"]["correct"])
    for key in (("all","all"),("family",row["family"]),("view",row["view"]),("language",row["language"]),("status",exp["status"])):
        tot[key]+=1; strict[key]+=ok_s; len_[key]+=ok_l; fmt[key]+= (ok_l and not ok_s)
out={}
for key in sorted(tot):
    out["/".join(key)]={"n":tot[key],"strict":round(strict[key]/tot[key],3),"lenient":round(len_[key]/tot[key],3),"format_only_failures":fmt[key]}
for k,v in out.items(): print(f"{k:45s} n={v['n']:4d} strict={v['strict']:.3f} lenient={v['lenient']:.3f} fmt_fail={v['format_only_failures']}")
if "--out" in sys.argv: json.dump(out,open(sys.argv[sys.argv.index("--out")+1],"w"),indent=1)
