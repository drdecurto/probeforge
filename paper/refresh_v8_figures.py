"""Regenerate Figures 7--10 from the included round15 snapshot; requires numpy and reportlab."""
from pathlib import Path
import json,csv,statistics as st,shutil,re
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
R=Path(__file__).resolve().parent; L=['en','es','ca','ar','lb','eu','fr','pt','it','de','ru','zh-cn','zh-tw']
A=json.load(open(R/'evidence/round15/energybench_analysis_all.json'))
K={(r['protocol'],r['slot']):float(r['mean']) for r in csv.DictReader(open(R/'evidence/round15/energybench_answer_accuracy.csv'))}
mean=lambda p,s:st.mean(A[f'{p}/{s}/{l}']['overall_average'] for l in L)
slots=['qwen3-8b-base','qwen3-8b-mined','qwen3-8b-v4','qwen3-8b-v4b','qwen3-14b-base','qwen3-14b-random','qwen3-14b-mixed50','phi-4-base','phi-4-mined','phi-4-mini-instruct','phi-4-mini-mined','phi-4-mini-random','phi-4-mini-v4','phi-4-mini-v4b']
labels={s:s.replace('14b','14B').replace('8b','8B').replace('qwen3-','Qwen3-').replace('phi-4-mini-instruct','Phi-4-mini base').replace('phi-4','Phi-4').replace('-base',' base').replace('-mined',' mined').replace('-random',' random').replace('-mixed50',' mixed50').replace('-v4b',' concise').replace('-v4',' diverse') for s in slots}
# Vector plots: equal axes, paired dots, explicit signed changes.
def chart(path,title,rows,xmin,xmax,xlabel,height=None):
    h=height or 58+24*len(rows);c=canvas.Canvas(str(R/'figures'/path),pagesize=(620,h)); left=205;right=560;bot=37;top=h-35
    c.setFont('Helvetica-Bold',14);c.drawString(8,h-15,title)
    x=lambda v:left+(v-xmin)/(xmax-xmin)*(right-left)
    for v in np.linspace(xmin,xmax,5):
        c.setStrokeColor(HexColor('#e4e7eb'));c.line(x(v),bot,x(v),top);c.setFillColor(HexColor('#555555'));c.setFont('Helvetica',11);c.drawCentredString(x(v),bot-12,f'{v:.2f}')
    for i,(label,b,a) in enumerate(rows):
        y=top-10-i*24;c.setFillColor(HexColor('#222222'));c.setFont('Helvetica',12);c.drawRightString(left-12,y-3,label)
        c.setStrokeColor(HexColor('#a5aab1'));c.setLineWidth(2);c.line(x(b),y,x(a),y)
        c.setFillColor(HexColor('#707780'));c.circle(x(b),y,3,fill=1,stroke=0)
        c.setFillColor(HexColor('#007c83') if a>=b else HexColor('#b64c34'));c.circle(x(a),y,3.6,fill=1,stroke=0)
        c.setFont('Helvetica',11);c.drawString(right+8,y-3,f'{a-b:+.3f}')
    c.setFillColor(HexColor('#333333'));c.setFont('Helvetica',12);c.drawCentredString((left+right)/2,9,xlabel);c.save()
pairs=[('qwen3-8b-mined','qwen3-8b-base'),('qwen3-8b-v4','qwen3-8b-base'),('qwen3-8b-v4b','qwen3-8b-base'),('qwen3-14b-random','qwen3-14b-base'),('qwen3-14b-mixed50','qwen3-14b-base'),('phi-4-mini-mined','phi-4-mini-instruct'),('phi-4-mini-random','phi-4-mini-instruct'),('phi-4-mini-v4','phi-4-mini-instruct'),('phi-4-mini-v4b','phi-4-mini-instruct')]
for p,fn in [('out','fig7_energy_benchmark.pdf'),('out_chat','fig8_energy_chat.pdf')]:
    pp=pairs if p=='out' else [z for z in pairs if z[0]!='qwen3-14b-mixed50']+[('phi-4-mined','phi-4-base')]
    chart(fn,('Raw completion' if p=='out' else 'Chat template')+' | matched base-to-adapter changes',[(labels[s],mean(p,b),mean(p,s)) for s,b in pp],0,.65,'Score | grey: base; colour: adapter; right: change')
# Complete eight-discipline results, four representative chat adapters, no category selection.
cats=list(A['out_chat/qwen3-8b-base/en']['category_averages']); dp=[('qwen3-8b-mined','qwen3-8b-base'),('qwen3-8b-v4b','qwen3-8b-base'),('qwen3-14b-random','qwen3-14b-base'),('phi-4-mined','phi-4-base'),('phi-4-mini-mined','phi-4-mini-instruct'),('phi-4-mini-v4b','phi-4-mini-instruct')]
from collections import Counter
counts=json.loads((R/'evidence/round15/category_counts.json').read_text())
c=canvas.Canvas(str(R/'figures/fig10_disciplines.pdf'),pagesize=(620,280));c.setFont('Helvetica-Bold',14);c.drawString(8,264,'Chat-protocol change by discipline | adapter minus its own base')
short=['8B mined','8B concise','14B random','Phi-4 mined','Mini mined','Mini concise']; cells=[]
for j,lab in enumerate(short):c.setFont('Helvetica-Bold',12);c.drawCentredString(170+j*75,238,lab)
for i,cat in enumerate(cats):
    y=216-i*24;c.setFillColor(HexColor('#222222'));c.setFont('Helvetica',12);c.drawRightString(127,y,cat+f' (n={counts[cat]})')
    for j,(s,b) in enumerate(dp):
        v=st.mean(A[f'out_chat/{s}/{l}']['category_averages'][cat]-A[f'out_chat/{b}/{l}']['category_averages'][cat] for l in L);cells.append({'discipline':cat,'roots':counts[cat],'adapter':s,'delta':v})
        c.setFillColor(HexColor('#d6eeeb') if v>=0 else HexColor('#f8dfd5'));c.rect(135+j*75,y-6,70,22,fill=1,stroke=0);c.setFillColor(HexColor('#174d50') if v>=0 else HexColor('#843520'));c.drawCentredString(170+j*75,y,f'{v:+.3f}')
c.setFillColor(HexColor('#444444'));c.setFont('Helvetica',11);c.drawString(8,10,'All eight categories; 13 languages; three repeats. Descriptive differences.');c.save()
with open(R/'source_tables/discipline_changes.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=cells[0]);w.writeheader();w.writerows(cells)
print('discipline',[(s,min(x['delta'] for x in cells if x['adapter']==s),max(x['delta'] for x in cells if x['adapter']==s)) for s,b in dp])

rows=list(csv.DictReader(open(R/'evidence/round15/stdbench_stdbench_summary.csv')));by={r['model']:r for r in rows}
order=[r['model'] for r in rows[:12]]
order=order[:9]+['qwen3-8b-v4','qwen3-8b-v4b']+order[9:]+['phi-4-mini-mined','phi-4-mini-random','phi-4-mini-v4','phi-4-mini-v4b']+[r['model'] for r in rows[12:18]]
def base(s):
    if s.startswith('qwen3-14b'):return 'qwen3-14b-base'
    if s.startswith('qwen3-8b'):return 'qwen3-8b-base'
    if s.startswith('phi-4-mini'):return 'phi-4-mini-instruct'
    if s.startswith('phi-4'):return 'phi-4-base'
    return s
name=lambda s:labels.get(s,s.replace('qwen3-','Qwen3-').replace('14b','14B').replace('8b','8B'))
chart('fig9_standard_benchmarks.pdf','MGSM strict extraction | change within each model family',[(name(s),float(by[base(s)]['mgsmstrict_mean']),float(by[s]['mgsmstrict_mean'])) for s in order if base(s)!=s],.65,.95,'Exact match | grey: base; colour: adapter; right: change')
