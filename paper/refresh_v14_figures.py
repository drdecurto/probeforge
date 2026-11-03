"""Refresh energy-transfer plots with replicated Phi-4 and Aya extensions."""
from pathlib import Path
import json,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;E=P/'evidence';F=P/'figures'
read=lambda p:json.loads(p.read_text())
A=read(E/'round15/energybench_analysis_all.json')
for n in [16,17,18,19]:
 for f in (E/f'round{n}/energybench').rglob('*.json'):A[str(f.relative_to(E/f'round{n}/energybench'))[:-5]]=read(f)['analysis']
for f in (E/'post_v13/energybench').rglob('*.json'):
 d=read(f)
 if 'analysis' in d:A[str(f.relative_to(E/'post_v13/energybench'))[:-5]]=d['analysis']
L=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
def energy(slot,proto):return st.mean(A[f'{proto}/{slot}/{l}']['overall_average'] for l in L)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
for proto,name in [('out','fig7_energy_benchmark'),('out_chat','fig8_energy_chat')]:
 pairs=[('8B mined',['qwen3-8b-mined'],'qwen3-8b-base'),('8B concise',['qwen3-8b-v4b'],'qwen3-8b-base'),('8B mixed-format (3)',[f'v4c-8b_s{s}' for s in [1,2,3]],'qwen3-8b-base'),('14B random',['qwen3-14b-random'],'qwen3-14b-base'),('14B mixed-format (3)',[f'v4c-14b_s{s}' for s in [1,2,3]],'qwen3-14b-base'),('Mini mixed-format (3)',[f'v4c-phi4mini_s{s}' for s in [1,2,3]],'phi-4-mini-instruct'),('Aya mined (3)',['aya-mined','aya-mined_s2','aya-mined_s3'],'aya-expanse-8b'),('Aya mixed-format (3)',[f'v4c-aya_own_s{s}' for s in [1,2,3]],'aya-expanse-8b')]
 if proto=='out_chat':pairs[5:5]=[('Phi-4 mined (3)',['phi-4-mined','phi-4-mined_s1','phi-4-mined_s2'],'phi-4-base'),('Phi-4 mixed-format (3)',[f'v4c-phi4_s{s}' for s in [1,2,3]],'phi-4-base')]
 fig,ax=plt.subplots(figsize=(6.4,3.1 if proto=='out_chat' else 2.9));labs=[]
 for i,(lab,slots,base) in enumerate(pairs):
  bv=energy(base,proto);v=[energy(s,proto) for s in slots];m=st.mean(v);sd=st.stdev(v) if len(v)>1 else 0;c='#007c83' if m>=bv else '#b64c34'
  ax.plot([bv,m],[i,i],color='#a5aab1',lw=1.5);ax.plot(bv,i,'o',color='#707780',ms=4);ax.errorbar(m,i,xerr=sd if len(v)>1 else None,fmt='o',color=c,ms=4,capsize=2);ax.text(.66,i,f'{m-bv:+.3f}',va='center',fontsize=8,color=c);labs.append(lab)
 ax.set_yticks(range(len(labs)),labs);ax.invert_yaxis();ax.set_xlim(.30,.65);ax.set_xlabel('Reference similarity | grey: base; colour: adapter');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(F/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
