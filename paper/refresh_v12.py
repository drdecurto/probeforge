"""Build v12 additions from archived rounds 16--19. Older selected-checkpoint tables remain unchanged."""
from pathlib import Path
import json,csv,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; E=P/'evidence'; F=P/'figures'
def read(f):return json.loads(f.read_text())
def rep(f):return read(f)['budgets']['512']
def stat(v):return st.mean(v),st.stdev(v) if len(v)>1 else 0
A=read(E/'round15/energybench_analysis_all.json')
for n in [16,17,18,19]:
 for f in (E/f'round{n}/energybench').rglob('*.json'):
  key=str(f.relative_to(E/f'round{n}/energybench'))[:-5];A[key]=read(f)['analysis']
L=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
def energy(slot,proto):return st.mean(A[f'{proto}/{slot}/{l}']['overall_average'] for l in L)
K={}
for f in [E/'round15/energybench_answer_accuracy.csv',E/'round16/energybench/answer_accuracy.csv',E/'round17/energybench/answer_accuracy.csv']:
 for r in csv.DictReader(f.open()):K[r['protocol'],r['slot']]=float(r['mean'])
for r in csv.DictReader((E/'table10_sig_energy.csv').open()):
 if r['metric']=='answer_acc':K[{'raw':'out','chat':'out_chat'}[r['protocol']],r['arm']]=float(r['score'])
# Recompute key-value matches from full archived responses; retain exact earlier audit values.
for r in read(P/'VALIDATION_ROUND15.json')['secondary_checks']:K[r['protocol'],r['model']]=r['key_value_match']
ns={};exec((E/'round15/answer_acc.py').read_text().split('rows = []')[0],ns)
raw={}
for n in [16,17,18,19]:
 for f in (E/f'round{n}/energybench').rglob('*.json'):raw[str(f.relative_to(E/f'round{n}/energybench'))[:-5]]=f
for proto,slot in sorted({tuple(k.split('/')[:2]) for k in raw}):
 values=[]
 for lang in L:
  d=read(raw[f'{proto}/{slot}/{lang}']);sol=d['solutions'];sol=list(sol.values()) if isinstance(sol,dict) else sol;hits=[]
  for prob,runs in zip(d['problems'],sol):
   key=ns['key_value'](prob['problem'],prob['solution'])
   if key is not None:hits.extend(any(ns['close'](v,key) for v in ns['vals'](run)) for run in runs)
  values.append(st.mean(hits))
 K[proto,slot]=st.mean(values)
S={}
for n in [16,17,18,19]:
 for r in csv.DictReader((E/f'round{n}/stdbench/stdbench_summary.csv').open()):S[r['model']]=r
# Main Phi-4 replication; update only the two rows.
phi={arm:[rep(E/f'round8/report12_{arm}.json')]+[rep(E/f'round19/evalphi4s/report12_{arm}_s{s}.json') for s in [1,2]] for arm in ['random','mined']}
f=P/'tables/main.tex';text=f.read_text()
for arm,rs in phi.items():
 m,sd=stat([r['pass_at_1'] for r in rs]);v=[st.mean(r['worst_language'] for r in rs),st.mean(r['per_view']['boundary'] for r in rs),st.mean(r['per_view']['counterfactual'] for r in rs)]
 row=f'Phi-4 & {arm} & 3 & {m:.3f} & {sd:.3f} & '+ ' & '.join(f'{x:.3f}' for x in v)+f" & {round(st.mean(r['mean_completion_tokens'] for r in rs))}"+r'\\'
 text='\n'.join(row if l.startswith(f'Phi-4 & {arm} &') else l for l in text.splitlines())+'\n'
f.write_text(text)
# Main plot uses exact report values.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(6.4,3.2));colors=['#8a8985','#1baf7a','#2a78d6','#eb6834']
for j,arm in enumerate(['base','translated','random','mined']):
 vals=[];errs=[]
 for n in [4,6,8]:
  if n==8 and arm in phi:rs=phi[arm]
  else:
   base='base14b' if n==6 and arm=='base' else arm
   paths=[E/f'round{n}/report12_{base}.json']
   if arm in ['random','mined'] and n!=8:paths +=[E/f'round{n}/report12_{arm}_s{s}.json' for s in [1,2]]
   rs=[rep(f) for f in paths]
  m,sd=stat([r['pass_at_1'] for r in rs]);vals.append(m);errs.append(sd)
 ax.bar([i+(j-1.5)*.19 for i in range(3)],vals,width=.18,yerr=errs,capsize=2,color=colors[j],label=arm)
ax.set_xticks(range(3),['Qwen3-8B','Qwen3-14B','Phi-4']);ax.set_ylabel('Exact pass@1');ax.set_ylim(0,.51);ax.legend(ncol=4,fontsize=8,frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(F/f'fig1_arms_by_model.{ext}',dpi=180)
plt.close(fig)
# Compact replication and boundary-of-transfer table.
groups=[('Qwen3-8B base',['qwen3-8b-base'],[E/'round4/report12_base.json']),('Qwen3-8B mixed-format',[f'v4c-8b_s{s}' for s in [1,2,3]],[E/f'round17/evalv4c/report12_8b_s{s}.json' for s in [1,2,3]]),('Qwen3-14B base',['qwen3-14b-base'],[E/'round6/report12_base14b.json']),('Qwen3-14B mixed-format',[f'v4c-14b_s{s}' for s in [1,2,3]],[E/f'round17/evalv4c/report12_14b_s{s}.json' for s in [1,2,3]]),('Phi-4 base',['phi-4-base'],[E/'round8/report12_base.json']),('Phi-4 mined',['phi-4-mined','phi-4-mined_s1','phi-4-mined_s2'],[E/'round8/report12_mined.json',E/'round19/evalphi4s/report12_mined_s1.json',E/'round19/evalphi4s/report12_mined_s2.json']),('Phi-4-mini base',['phi-4-mini-instruct'],[E/'round15/evalphi4mini_report12_base.json']),('Phi-4-mini mixed-format',[f'v4c-phi4mini_s{s}' for s in [1,2,3]],[E/f'round17/evalv4c/report12_phi4mini_s{s}.json' for s in [1,2,3]]),('Aya-8B base',['aya-expanse-8b'],[E/'round18/evalaya/report12_base.json']),('Aya-8B mined',['aya-mined'],[E/'round18/evalaya/report12_mined.json']),('Aya-8B mixed-format',['v4c-aya_own_s1'],[E/'round18/evalv4c/report12_aya_own_s1.json'])]
rows=[]
for label,slots,paths in groups:
 d={'label':label,'seeds':len(slots),'main':stat([rep(f)['pass_at_1'] for f in paths])}
 for proto in ['out','out_chat']:
  d[proto]=None if label.startswith('Phi-4 ') and proto=='out' else stat([energy(s,proto) for s in slots])
 d['key']=stat([K['out_chat',s] for s in slots])
 for metric in ['bele_mean','gmmlu_mean','mgsmstrict_mean']:d[metric]=stat([float(S[s][metric]) for s in slots])
 rows.append(d)
fmt=lambda v,n: '---' if v is None else (f'{v[0]:.3f}' if n==1 else f'${v[0]:.3f}\\pm{v[1]:.3f}$')
t=[r'\begin{table*}[t]',r'\caption{Replicated transfer and cross-family limits. Mixed-format denotes the v4c mixture; Aya uses its own mined curriculum. Entries are means $\pm$ sample SD for three seeds, or one evaluation where $n=1$. S is whole-response reference similarity; K is secondary key-value match, not exact answer accuracy. MGSM uses strict extraction. Base rows provide matched references; bold means exceed the corresponding base after rounding (not a significance marker).}\label{tab:replicated}',r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrccccccc}\toprule',r'Checkpoint & $n$ & ProbeForge & Raw S & Chat S & Chat K & Belebele & GMMLU & MGSM\\\midrule']
for d in rows:
 if d['label'].endswith('base'):base=d
 cells=[]
 for k in ['main','out','out_chat','key','bele_mean','gmmlu_mean','mgsmstrict_mean']:
  cell=fmt(d[k],d['seeds'])
  if d is not base and d[k] is not None and base[k] is not None and round(d[k][0],3)>round(base[k][0],3):
   cell=(f'$\\mathbf{{{d[k][0]:.3f}}}\\pm{d[k][1]:.3f}$' if d['seeds']>1 else r'\textbf{'+cell+'}')
  cells.append(cell)
 t.append(d['label']+' & '+str(d['seeds'])+' & '+' & '.join(cells)+r'\\')
t +=[r'\bottomrule\end{tabular}',r'\end{table*}'];(P/'tables/replicated.tex').write_text('\n'.join(t)+'\n');(E/'v11_aggregates.json').write_text(json.dumps(rows,indent=2)+'\n')
# Paired transfer plots: include historical concise arm, replicated mixed-format arms and Aya.
for proto,filename in [('out','fig7_energy_benchmark'),('out_chat','fig8_energy_chat')]:
 pairs=[('8B mined',['qwen3-8b-mined'],'qwen3-8b-base'),('8B concise',['qwen3-8b-v4b'],'qwen3-8b-base'),('8B mixed-format (3)',[f'v4c-8b_s{s}' for s in [1,2,3]],'qwen3-8b-base'),('14B random',['qwen3-14b-random'],'qwen3-14b-base'),('14B mixed-format (3)',[f'v4c-14b_s{s}' for s in [1,2,3]],'qwen3-14b-base'),('Mini mixed-format (3)',[f'v4c-phi4mini_s{s}' for s in [1,2,3]],'phi-4-mini-instruct'),('Aya mined',['aya-mined'],'aya-expanse-8b'),('Aya mixed-format',['v4c-aya_own_s1'],'aya-expanse-8b')]
 if proto=='out_chat':pairs.insert(5,('Phi-4 mined (3)',['phi-4-mined','phi-4-mined_s1','phi-4-mined_s2'],'phi-4-base'))
 fig,ax=plt.subplots(figsize=(6.4,2.9));labs=[]
 for i,(lab,ss,b) in enumerate(pairs):
  bv=energy(b,proto);m,sd=stat([energy(s,proto) for s in ss]);color='#007c83' if m>=bv else '#b64c34'
  ax.plot([bv,m],[i,i],color='#a5aab1',lw=1.5);ax.plot(bv,i,'o',color='#707780',ms=4);ax.errorbar(m,i,xerr=sd if len(ss)>1 else None,fmt='o',color=color,ms=4,capsize=2);ax.text(.66,i,f'{m-bv:+.3f}',va='center',fontsize=8,color=color);labs.append(lab)
 ax.set_yticks(range(len(labs)),labs);ax.invert_yaxis();ax.set_xlim(.30,.65);ax.set_xlabel('Reference similarity | grey: base; colour: adapter');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(F/f'{filename}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
print(json.dumps(rows,indent=2))
