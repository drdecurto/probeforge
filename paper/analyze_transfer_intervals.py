"""Reproduce paired problem-bootstrap intervals from archived scores (no inference)."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent
E=P/'evidence'
langs=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
read=lambda p:json.loads(p.read_text())
results=[]
for model in ['8b','14b']:
 for protocol in ['out','out_chat']:
  diffs=[]
  for lang in langs:
   base=read(E/'transfer_intervals'/protocol/f'qwen3-{model}-base'/f'{lang}.json')
   seeds=[]
   for seed in [1,2,3]:
    paths=[E/f'round{r}'/'energybench'/protocol/f'v4c-{model}_s{seed}'/f'{lang}.json' for r in [16,17]]
    paths=[p for p in paths if p.exists()]
    assert len(paths)==1,paths
    derived=read(paths[0]);assert derived['problems']==base['problems']
    assert len(derived['average_scores'])==len(base['average_scores'])==79
    seeds.append(derived['average_scores'])
   diffs.append(np.mean(seeds,axis=0)-np.array(base['average_scores']))
  per_problem=np.mean(diffs,axis=0)
  rng=np.random.default_rng(20261028)
  draws=rng.integers(0,79,size=(10000,79))
  low,high=np.quantile(per_problem[draws].mean(axis=1),[.025,.975])
  results.append(dict(model=model,protocol=protocol,delta=float(per_problem.mean()),low=float(low),high=float(high)))
expected=read(E/'transfer_intervals/intervals.json')
for a,b in zip(results,expected['results']):
 assert a['model']==b['model'] and a['protocol']==b['protocol']
 for k in ['delta','low','high']:assert abs(a[k]-b[k])<1e-12,(k,a[k],b[k])
print(json.dumps(results,indent=2))
