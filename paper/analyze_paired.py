"""Recompute paired correctness from complete archived main-test responses (no inference)."""
from pathlib import Path
import json,hashlib,statistics as st
P=Path(__file__).resolve().parent;E=P/'evidence/paired'
read=lambda f:[json.loads(l) for l in f.read_text().splitlines()]
rows=read(E/'test_eval12.jsonl');by={r['id']:r for r in rows};views={'base','paraphrase','distractor','context_academic','context_business','expansion'};out=[]
for f in sorted(E.glob('r*_test_*.jsonl')):
 recs=read(f);assert len(recs)==len({r['id'] for r in recs})==1248;cells={};langs={}
 for r in recs:
  d=by[r['id']];assert r['example_hash']==hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  assert r['group_id']==d['group_id'] and r['budget']==512 and r['sample']==0 and not r['synthetic']
  ok=bool(r['verification']['correct']) and r['finish_reason']=='stop'
  if d['view'] in views:cells.setdefault((d['group_id'],d['language']),[]).append(ok)
  if d['view']=='base':langs.setdefault(d['group_id'],[]).append(ok)
 assert len(cells)==156 and all(len(v)==6 for v in cells.values())
 assert len(langs)==12 and all(len(v)==13 for v in langs.values())
 result=dict(file=f.name,pass_at_1=sum(r['verification']['correct'] for r in recs)/1248,all_six=sum(all(v) for v in cells.values()),all_languages_base=sum(all(v) for v in langs.values()))
 n=int(f.name.split('_')[0][1:]);arm=f.stem.replace(f'r{n}_test_','');rn='base14b' if n==6 and arm=='base' else arm
 expected=json.loads((P/f'evidence/round{n}/report12_{rn}.json').read_text())['budgets']['512']['pass_at_1'];assert abs(result['pass_at_1']-expected)<1e-12
 out.append(result)
extensions=[]
for family,model,slots in [('q32','Qwen3-32B',['base32b','random32b','v4c32b']),('glm','GLM-4.7-Flash',['glmbase','glmrandom','glmv4c'])]:
 for arm,slot in zip(['base','random','mixed-format'],slots):
  f=P/f'evidence/round20/{family}/eval/test_{slot}.jsonl';recs=read(f)
  assert len(recs)==len({r['id'] for r in recs})==1248
  cells={};langs={}
  for r in recs:
   d=by[r['id']]
   assert r['example_hash']==hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
   assert r['group_id']==d['group_id'] and r['budget']==512 and r['sample']==0 and not r['synthetic']
   ok=bool(r['verification']['correct']) and r['finish_reason']=='stop'
   if d['view'] in views:cells.setdefault((d['group_id'],d['language']),[]).append(ok)
   if d['view']=='base':langs.setdefault(d['group_id'],[]).append(ok)
  assert len(cells)==156 and all(len(v)==6 for v in cells.values())
  assert len(langs)==12 and all(len(v)==13 for v in langs.values())
  result=dict(file=str(f.relative_to(P)),pass_at_1=sum(r['verification']['correct'] for r in recs)/1248,all_six=sum(all(v) for v in cells.values()),all_languages_base=sum(all(v) for v in langs.values()))
  expected=json.loads((f.parent/f'report12_{slot}.json').read_text())['budgets']['512']['pass_at_1']
  assert abs(result['pass_at_1']-expected)<1e-12
  out.append(result);extensions.append((model+' '+arm,[result]))
(E/'paired_results.json').write_text(json.dumps(out,indent=2)+'\n')
groups=[('Qwen3-8B base',[r for r in out if r['file']=='r4_test_base.jsonl']),('Qwen3-8B mined',[r for r in out if r['file'].startswith('r4_test_mined')]),('Qwen3-14B base',[r for r in out if r['file']=='r6_test_base.jsonl']),('Qwen3-14B random',[r for r in out if r['file'].startswith('r6_test_random')]),('Phi-4 base',[r for r in out if r['file']=='r8_test_base.jsonl']),('Phi-4 mined',[r for r in out if r['file']=='r8_test_mined.jsonl' or r['file'].startswith('r19_test_mined')])]
groups += extensions
t=[r'\begin{table}[tbp]',r'\caption{Paired correctness on the main test. Six-view correctness requires the base and five answer-preserving transformations to pass jointly (156 root--language cells). Cross-language correctness requires the base view to pass in all 13 language configurations (12 roots). The 8B, 14B and Phi-4 adapters report three-seed means $\pm$ sample SD; the 32B and GLM extensions use one seed. Base rows are single evaluations.}\label{tab:paired}',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lcc}\toprule',r'Checkpoint & Six views & All languages\\\midrule']
for name,rs in groups:
 cells=[]
 for k,n in [('all_six',156),('all_languages_base',12)]:
  vs=[r[k]/n for r in rs];cells.append(f'{st.mean(vs):.3f}' if len(vs)==1 else f'${st.mean(vs):.3f}\\pm{st.stdev(vs):.3f}$')
 t.append(name+' & '+' & '.join(cells)+r'\\')
t +=[r'\bottomrule\end{tabular}',r'\end{table}'];(P/'tables/paired.tex').write_text('\n'.join(t)+'\n')
print('\n'.join(t))
