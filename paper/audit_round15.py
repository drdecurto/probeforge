"""Usage: python audit_round15.py /path/to/extracted/round15 (requires numpy)."""
from pathlib import Path
import json,csv,statistics as st,shutil,re
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
import sys
R=Path(__file__).resolve().parent; E=Path(sys.argv[1])/'energybench'; L=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
A=json.load(open(R/'evidence/round15/energybench_analysis_all.json'))
K={(r['protocol'],r['slot']):float(r['mean']) for r in csv.DictReader(open(R/'evidence/round15/energybench_answer_accuracy.csv'))}
mean=lambda p,s:st.mean(A[f'{p}/{s}/{l}']['overall_average'] for l in L)
slots=['qwen3-8b-base','qwen3-8b-mined','qwen3-8b-v4','qwen3-8b-v4b','qwen3-14b-base','qwen3-14b-random','qwen3-14b-mixed50','phi-4-base','phi-4-mined','phi-4-mini-instruct','phi-4-mini-mined','phi-4-mini-random','phi-4-mini-v4','phi-4-mini-v4b']
labels={s:s.replace('qwen3-','Qwen3-').replace('phi-4-mini-instruct','Phi-4-mini base').replace('phi-4','Phi-4').replace('-base',' base').replace('-mined',' mined').replace('-random',' random').replace('-mixed50',' mixed50').replace('-v4b',' concise').replace('-v4',' diverse') for s in slots}
# Recompute secondary measure using archived scorer (execute definitions only).
ns={};exec((E/'answer_acc.py').read_text().split('rows = []')[0],ns)
checks=[]; cache={}
for p,s in K:
    vals=[]
    for l in L:
        d=json.load(open(E/p/s/(l+'.json'))); sol=d['solutions'];sol=list(sol.values()) if isinstance(sol,dict) else sol
        hits=[]
        for prob,runs in zip(d['problems'],sol):
            kv=ns['key_value'](prob['problem'],prob['solution'])
            if kv is not None: hits.extend(any(ns['close'](v,kv) for v in ns['vals'](r)) for r in runs)
        vals.append(st.mean(hits))
        cache[p,s,l]=d
    assert round(st.mean(vals),3)==K[p,s],(p,s)
    checks.append({'protocol':p,'model':s,'key_value_match':st.mean(vals)})
# All raw/chat scores checked against individual files.
for p,s,l in cache:
    d=cache[p,s,l]
    assert abs(st.mean(d['average_scores'])-A[f'{p}/{s}/{l}']['overall_average'])<1e-9
# Problem-cluster paired intervals: same 79 indices across languages.
rng=np.random.default_rng(20261023); ix=rng.integers(0,79,(10000,79)); ci={}
for p in ['out','out_chat']:
    for s,b in [('qwen3-8b-v4b','qwen3-8b-base'),('phi-4-mini-v4b','phi-4-mini-instruct')]:
        delta=np.mean([np.array(cache[p,s,l]['average_scores'])-cache[p,b,l]['average_scores'] for l in L],axis=0)
        lo,hi=np.quantile(delta[ix].mean(axis=1),[.025,.975]);ci[p,s]=[float(delta.mean()),float(lo),float(hi)]
print('Intervals',ci)
(R/'VALIDATION_ROUND15.json').write_text(json.dumps({'secondary_checks':checks,'primary_verified_files':len(cache),'bootstrap':{'replicates':10000,'seed':20261023,'cluster':'problem index across all languages and three responses','intervals':{str(k):v for k,v in ci.items()}}},indent=2))
