"""Validate completed round-20 main responses and regenerate the extension table."""
from pathlib import Path
import json,csv,hashlib,statistics as st
P=Path(__file__).resolve().parent; E=P/'evidence/round20'
read=lambda p:json.loads(p.read_text())
L=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
by={r['id']:r for r in map(json.loads,(P/'evidence/paired/test_eval12.jsonl').read_text().splitlines())}
S={r['model']:r for r in csv.DictReader((E/'stdbench/stdbench_summary.csv').open())}
ns={};exec((P/'evidence/round15/answer_acc.py').read_text().split('rows = []')[0],ns)
out=[]
for family,prefix,slots,label in [('q32','qwen3-32b',['base32b','random32b','v4c32b'],'Qwen3-32B'),('glm','glm47flash',['glmbase','glmrandom','glmv4c'],'GLM-4.7-Flash')]:
 for arm,slot in zip(['base','random','v4c'],slots):
  report=read(E/f'{family}/eval/report12_{slot}.json');d=report['budgets']['512']
  rr=list(map(json.loads,(E/f'{family}/eval/test_{slot}.jsonl').read_text().splitlines()))
  assert len(rr)==len({r['id'] for r in rr})==1248 and d['missing_requests']==0
  for r in rr:
   expected=hashlib.sha256(json.dumps(by[r['id']],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
   assert expected==r['example_hash'] and r['budget']==512 and not r['synthetic']
  assert abs(st.mean(r['verification']['correct'] for r in rr)-d['pass_at_1'])<1e-12
  a=dict(model=label,arm=arm,main=d['pass_at_1'],worst=d['worst_language'],tokens=d['mean_completion_tokens'])
  for proto in ['out','out_chat']:
   scores=[];hits=[]
   for lang in L:
    z=read(E/f'{family}/energy/{proto}/{slot}/{lang}.json');b=read(E/f'{family}/energy/{proto}/{slots[0]}/{lang}.json')
    assert z['problems']==b['problems'] and len(z['average_scores'])==79
    sol=z['solutions'];sol=list(sol.values()) if isinstance(sol,dict) else sol
    assert len(sol)==79 and all(len(r)==3 for r in sol)
    scores.append(st.mean(z['average_scores']));h=[]
    for prob,runs in zip(z['problems'],sol):
     key=ns['key_value'](prob['problem'],prob['solution'])
     if key is not None:h.extend(any(ns['close'](v,key) for v in ns['vals'](r)) for r in runs)
    hits.append(st.mean(h))
   a[proto]=st.mean(scores);a[proto+'_key']=st.mean(hits)
  s=S[prefix+'-'+arm]
  for key,start in [('bele','bele_'),('gmmlu','gmmlu_'),('mgsm','mgsmstrict_'),('flex','mgsm_')]:
   a[key]=st.mean(float(v) for k,v in s.items() if k.startswith(start) and not k.endswith('mean'))
  out.append(a)
(P/'VALIDATION_ROUND20.json').write_text(json.dumps(out,indent=2)+'\n')
t=[r'\begin{table*}[t]',r'\caption{Single-seed scale and architecture extensions (seed 1). Main is exact accuracy on the same 1248-prompt test; raw/chat S and chat K follow Table~\ref{tab:replicated}. MGSM uses strict extraction. Bold values exceed the matched base after rounding, not a significance threshold. Quantization and adapter targets differ across families.}\label{tab:scale}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{llrrrrrrr}\toprule',r'Model & Curriculum & Main & Raw S & Chat S & Chat K & Belebele & GMMLU & MGSM\\\midrule']
for a in out:
 if a['arm']=='base':base=a
 vals=[]
 for k in ['main','out','out_chat','out_chat_key','bele','gmmlu','mgsm']:
  v=f"{a[k]:.3f}";vals.append(r'\textbf{'+v+'}' if a['arm']!='base' and round(a[k],3)>round(base[k],3) else v)
 t.append(a['model']+' & '+{'base':'Base','random':'Random','v4c':'Mixed-format'}[a['arm']]+' & '+' & '.join(vals)+r'\\')
t += [r'\bottomrule\end{tabular}',r'\end{table*}']
(P/'tables/scale.tex').write_text('\n'.join(t)+'\n')
print(json.dumps(out,indent=2))
# Cross-check language aggregates against the harness result files, not rounded CSV means.
for model,s in S.items():
 folder=E/'stdbench/out'/model
 if not folder.exists():continue
 results={}
 for f in folder.rglob('results_*.json'):results.update(read(f)['results'])
 for lang in ['de','en','es','fr','ru','zh']:
  r=results['mgsm_native_cot_'+lang]
  assert abs(r['exact_match,strict-match']-float(s['mgsmstrict_'+lang]))<1e-12
  assert abs(r['exact_match,flexible-extract']-float(s['mgsm_'+lang]))<1e-12
 for k,v in s.items():
  if k.startswith('bele_') and k!='bele_mean':assert abs(results['belebele_'+k[5:]]['acc,none']-float(v))<1e-12
  if k.startswith('gmmlu_') and k!='gmmlu_mean':assert abs(results['global_mmlu_'+k[6:]]['acc,none']-float(v))<1e-12
print('All six harness summaries agree with archived per-language results.')
