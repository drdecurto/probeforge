"""Recompute post-v13 tables and diagnostics from archived evidence; no inference."""
from pathlib import Path
import json,csv,hashlib,statistics as st,random
import numpy as np
P=Path(__file__).resolve().parent; E=P/'evidence'; N=E/'post_v13'
read=lambda p:json.loads(p.read_text())
L=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
stat=lambda a:[st.mean(a),st.stdev(a) if len(a)>1 else 0]
by={r['id']:r for r in map(json.loads,(E/'paired/test_eval12.jsonl').read_text().splitlines())}
S={}
for f in [N/'stdbench/stdbench_summary.csv',N/'stdbench/stdbench_kol.csv']:
 for r in csv.DictReader(f.open()):S[r['model']]=r
ns={};exec((N/'energybench/answer_acc.py').read_text().split('rows = []')[0],ns)
EC={}; mainchecks={}; intervals=[]
def main(f):
 d=read(f)['budgets']['512'];test=f.with_name(f.name.replace('report12_','test_')).with_suffix('.jsonl')
 if test.exists():
  rs=list(map(json.loads,test.read_text().splitlines()));assert len(rs)==len({r['id'] for r in rs})==1248
  cells={};langs={}
  for r in rs:
   src=by[r['id']];assert r['example_hash']==hashlib.sha256(json.dumps(src,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
   assert r['budget']==512 and not r['synthetic'] and r['sample']==0
   ok=bool(r['verification']['correct']) and r['finish_reason']=='stop'
   if src['view'] in ['base','paraphrase','distractor','context_academic','context_business','expansion']:cells.setdefault((src['group_id'],src['language']),[]).append(ok)
   if src['view']=='base':langs.setdefault(src['group_id'],[]).append(ok)
  assert len(cells)==156 and all(len(x)==6 for x in cells.values());assert len(langs)==12 and all(len(x)==13 for x in langs.values())
  assert abs(st.mean(r['verification']['correct'] for r in rs)-d['pass_at_1'])<1e-12
  d['six']=st.mean(all(x) for x in cells.values());d['all_languages']=st.mean(all(x) for x in langs.values());mainchecks[str(f.relative_to(P))]={'six':d['six'],'all_languages':d['all_languages'],'length_stops':d['length_stops']}
 return d

def energy(root,slot,proto):
 key=str(root),slot,proto
 if key in EC:return EC[key]
 sem={};acc={};problems={}
 for l in L:
  d=read(root/proto/slot/f'{l}.json');a=d['average_scores'];a=list(a.values()) if isinstance(a,dict) else a;sol=d['solutions'];sol=list(sol.values()) if isinstance(sol,dict) else sol
  assert len(a)==len(sol)==len(d['problems'])==79 and all(len(r)==3 for r in sol)
  for i,(prob,runs,x) in enumerate(zip(d['problems'],sol,a)):
   if x is not None:sem[l,i]=x
   kv=ns['key_value'](prob['problem'],prob['solution'])
   if kv is not None:acc[l,i]=st.mean(any(ns['close'](v,kv) for v in ns['vals'](r)) for r in runs)
   problems[l,i]=prob
 EC[key]=(sem,acc,problems);return EC[key]

def boot(a,b):
 assert a.keys()==b.keys();ids=sorted({i for _,i in a});diff={i:[a[k]-b[k] for k in sorted(a) if k[1]==i] for i in ids}
 sums=np.array([sum(diff[i]) for i in ids]);counts=np.array([len(diff[i]) for i in ids]);rng=random.Random(1)
 idx=np.array([[rng.choice(range(len(ids))) for _ in ids] for _ in range(2000)])
 vals=np.sort(sums[idx].sum(1)/counts[idx].sum(1));return [float(sums.sum()/counts.sum()),float(vals[50]),float(vals[1949])]

def standard(slot):
 r=S[slot];out={}
 for key,prefix in [('bele','bele_'),('gmmlu','gmmlu_'),('mgsm','mgsmstrict_'),('flex','mgsm_')]:out[key]=st.mean(float(v) for k,v in r.items() if k.startswith(prefix) and not k.endswith('_mean'))
 files=list((N/'stdbench/out'/slot).rglob('results_*.json'));res={}
 for f in files:res.update(read(f)['results'])
 if files:
  for k,v in r.items():
   if k.startswith('bele_') and k!='bele_mean':assert abs(res['belebele_'+k[5:]]['acc,none']-float(v))<1e-12
   if k.startswith('gmmlu_') and k!='gmmlu_mean':assert abs(res['global_mmlu_'+k[6:]]['acc,none']-float(v))<1e-12
   if k.startswith('mgsmstrict_') and k!='mgsmstrict_mean':assert abs(res['mgsm_native_cot_'+k[11:]]['exact_match,strict-match']-float(v))<1e-12
 return out

rows=[]
def group(model,arm,reportfiles,energyroot,slots,stdslots,baseenergy=None):
 reports=[main(f) for f in reportfiles];d={'model':model,'arm':arm,'n':len(slots),'seeds':[]}
 for r,slot,stdslot in zip(reports,slots,stdslots):
  a={'main':r['pass_at_1'],'worst':r['worst_language'],'tokens':r['mean_completion_tokens'],'length_stops':r['length_stops'],'languages':r['per_language'],'views':r['per_view']};a.update(standard(stdslot))
  if 'six' in r:a.update(six=r['six'],all_languages=r['all_languages'])
  for proto in ['out','out_chat']:
   es=energy(energyroot,slot,proto);a[proto]=st.mean(es[0].values());a[proto+'_key']=st.mean(es[1].values())
   if baseenergy:
    bs=energy(baseenergy[0],baseenergy[1],proto);assert es[2]==bs[2]
    intervals.append({'model':model,'arm':arm,'slot':slot,'protocol':proto,'S':boot(es[0],bs[0]),'K':boot(es[1],bs[1])})
  d['seeds'].append(a)
 for k in ['main','worst','tokens','length_stops','out','out_chat','out_key','out_chat_key','bele','gmmlu','mgsm','flex','six','all_languages']:
  if k in d['seeds'][0]:d[k]=stat([a[k] for a in d['seeds']])
 rows.append(d);return d

for model,fam,stem,stdprefix in [('Qwen3-32B','q32','32b','qwen3-32b'),('GLM-4.7-Flash','glm','glm','glm47flash'),('Kolibri-1','kolibri','kol','kol-')]:
 root=N/fam;base={'q32':'base32b','glm':'glmbase','kolibri':'kolbase'}[fam];base_std=stdprefix+'-base' if fam!='kolibri' else 'kol-kolbase'
 group(model,'base',[root/'eval'/f'report12_{base}.json'],root/'energy',[base],[base_std])
 for arm in ['random','v4c']:
  slots=([arm+'32b']+[arm+'32b_s'+str(s) for s in [2,3]]) if fam=='q32' else ([stem+arm]+[stem+arm+'_s'+str(s) for s in [2,3]]) if fam=='glm' else [stem+arm+'_s'+str(s) for s in [1,2,3]]
  stdslots=[stdprefix+'-'+arm]+[stdprefix+'-'+arm+'_s'+str(s) for s in [2,3]] if fam!='kolibri' else ['kol-'+slot for slot in slots]
  group(model,arm,[root/'eval'/f'report12_{slot}.json' for slot in slots],root/'energy',slots,stdslots,(root/'energy',base))
# New Phi-4 and Aya replication; base Phi-4 raw scores remain excluded from manuscript tables.
group('Phi-4','v4c',[N/'evalr4'/f'report12_v4c-phi4_s{s}.json' for s in [1,2,3]],N/'energybench',[f'v4c-phi4_s{s}' for s in [1,2,3]],[f'v4c-phi4_s{s}' for s in [1,2,3]],(N/'energybench','phi-4-base'))
for arm in ['mined','v4c']:
 slots=['aya-mined','aya-mined_s2','aya-mined_s3'] if arm=='mined' else [f'v4c-aya_own_s{s}' for s in [1,2,3]]
 first=E/'round18'/('evalaya/report12_mined.json' if arm=='mined' else 'evalv4c/report12_aya_own_s1.json')
 group('Aya-8B',arm,[first]+[N/'evalr4'/f'report12_{slot}.json' for slot in slots[1:]],N/'energybench',slots,slots,(E/'round17/energybench','aya-expanse-8b'))
# Validate archived main paired deltas against independently verified report means.
for fam in ['q32','glm','kolibri']:
 for f in (N/fam/'eval').glob('cmp*.json'):
  c=read(f);base=read(f.parent/Path(c['baseline']).name)['budgets']['512']['pass_at_1']
  for x in c['comparisons']:
   y=read(f.parent/Path(x['treatment']).name)['budgets']['512']['pass_at_1'];assert abs(y-base-x['mean_paired_group_difference'])<1e-12
(P/'VALIDATION_V14_RESULTS.json').write_text(json.dumps({'rows':rows,'main_checks':mainchecks,'energy_intervals':intervals},indent=2)+'\n')
print('\n'.join(f"{r['model']} {r['arm']}: main={r['main']}, six={r.get('six')}, all_lang={r.get('all_languages')}, chatK={r['out_chat_key']}, std={[r[x] for x in ['bele','gmmlu','mgsm']]}" for r in rows))
