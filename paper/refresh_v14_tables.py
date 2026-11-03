"""Typeset validated v14 aggregates without changing manuscript prose."""
from pathlib import Path
import json,statistics as st,re
P=Path(__file__).resolve().parent;D=json.loads((P/'VALIDATION_V14_RESULTS.json').read_text());R=D['rows'];keys=['main','out','out_chat','out_chat_key','bele','gmmlu','mgsm']
def cell(v,n=3,b=None):
 m,s=v; t=f'{m:.3f}'
 if b is not None and round(m,3)>round(b,3):t=r'\mathbf{'+t+'}'
 return '$'+t+(rf'\pm{s:.3f}' if n>1 else '')+'$'
def line(r,b):return ' & '.join(cell(r[k],r['n'],b[k][0]) if not (r['model']=='Phi-4' and k=='out') else '---' for k in keys)+r'\\'
t=[r'\begin{table*}[t]',r'\caption{Three-seed scale and architecture extensions. Adapted rows give means $\pm$ sample SD; bases are single evaluations. Main uses the same 1248 prompts; S and K follow Table~\ref{tab:replicated}, and MGSM uses strict extraction. Bold means exceed the matched base after rounding, not a significance threshold. Adaptation implementations differ; Kolibri also uses a different standard-benchmark backend.}\label{tab:scale}',r'\centering\scriptsize\setlength{\tabcolsep}{2.5pt}',r'\begin{tabular}{llccccccc}\toprule',r'Model & Curriculum & Main & Raw S & Chat S & Chat K & Belebele & GMMLU & MGSM\\\midrule']
for r in R[:9]:
 if r['arm']=='base':b=r
 t.append(r['model']+' & '+{'base':'Base','random':'Random','v4c':'Mixed-format'}[r['arm']]+' & '+line(r,b))
t +=[r'\bottomrule\end{tabular}',r'\end{table*}'];(P/'tables/scale.tex').write_text('\n'.join(t)+'\n')
p=P/'tables/paired.tex';s=p.read_text();s=s.replace('The 8B, 14B and Phi-4 adapters report three-seed means $\\pm$ sample SD; the 32B and GLM extensions use one seed.','All adapted rows report three-seed means $\\pm$ sample SD.')
s=s[:s.index('Qwen3-32B base')]
for r in R[:9]:s+=r['model']+' '+{'base':'base','random':'random','v4c':'mixed-format'}[r['arm']]+' & '+cell(r['six'],r['n'])+' & '+cell(r['all_languages'],r['n'])+r'\\'+'\n'
s+=r'\bottomrule\end{tabular}'+'\n'+r'\end{table}'+'\n';p.write_text(s)
p=P/'tables/replicated.tex';s=p.read_text();bphi={'main':[.022],'out':[0],'out_chat':[.424],'out_chat_key':[.374],'bele':[.838],'gmmlu':[.705],'mgsm':[.787]};baya={'main':[0],'out':[.488],'out_chat':[.479],'out_chat_key':[.477],'bele':[.737],'gmmlu':[.540],'mgsm':[.707]}
for r in R[9:]:
 label=r['model']+' '+{'v4c':'mixed-format','mined':'mined'}[r['arm']];new=label+' & 3 & '+line(r,bphi if r['model']=='Phi-4' else baya)
 if r['model']=='Phi-4':
  s='\n'.join(l for l in s.splitlines() if not l.startswith(label+' &'))+'\n'
  s=s.replace('Phi-4-mini base',new+'\nPhi-4-mini base')
 else:s='\n'.join(new if l.startswith(label+' &') else l for l in s.splitlines())+'\n'
p.write_text(s)
t=[r'\begin{table}[tbp]',r'\caption{Kolibri-1 main-test accuracy by language (96 prompts each). Adapted columns average three seeds. All languages improve over base; the entries are descriptive, not independent language-level significance tests.}\label{tab:kolibri-languages}',r'\centering\footnotesize\setlength{\tabcolsep}{7pt}',r'\begin{tabular}{lccc}\toprule',r'Language & Base & Random & Mixed-format\\\midrule']
for l in ['en','de','es','ca','ar','lb','eu','fr','pt','it','ru','zh-cn','zh-tw']:
 t.append(l+' & '+' & '.join(f"{st.mean(a['languages'][l] for a in r['seeds']):.3f}" for r in R[6:9])+r'\\')
t +=[r'\bottomrule\end{tabular}',r'\end{table}'];(P/'tables/kolibri_languages.tex').write_text('\n'.join(t)+'\n')
