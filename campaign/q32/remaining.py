"""usage: remaining.py NAME INPUTS N  -> writes NAME_r{k}.json shards with ids not yet present in any test_NAME*.jsonl; prints remaining count"""
import json, sys, glob
name, inp, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); E = "/gpfs/scratch/bsc21/bsc021885/cf/q32/eval"
ids = [json.loads(l)["id"] for l in open(inp)]
have = set()
for f in glob.glob(f"{E}/test_{name}.*jsonl") + glob.glob(f"{E}/test_{name}.jsonl"):
    if f.endswith(".manifest.json"): continue
    for l in open(f):
        try: have.add(json.loads(l)["id"])
        except Exception: pass
rem = [i for i in ids if i not in have]
for old in glob.glob(f"{E}/{name}_r*.json"):
    import os; os.remove(old)
for k in range(n):
    part = rem[k::n]
    if part: json.dump(part, open(f"{E}/{name}_r{k}.json", "w"))
print(len(rem))
