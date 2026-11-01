"""Rebuild the categorical language plot from the v10 exact-value snapshot."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
series=json.loads((P/'evidence/language_plot_exact.json').read_text())
langs=list(series[0]['values'])
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
fig,ax=plt.subplots(figsize=(6.4,3.0))
for j,s in enumerate(series):
 xs=[i+(j-2)*0.12 for i in range(len(langs))]
 ax.plot(xs,[s['values'][l] for l in langs],linestyle='none',marker=s['marker'],ms=4.5,color=s['color'],markerfacecolor='white' if s['base'] else s['color'],label=s['name'])
ax.set_xticks(range(len(langs)),langs);ax.set_ylim(0,0.55);ax.set_ylabel('pass@1')
ax.grid(axis='y',color='#e6e5e1',linewidth=0.6)
ax.legend(frameon=False,fontsize=7.5,ncol=2)
ax.set_title('Per-language accuracy on the main benchmark',fontsize=9)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(P/'figures'/f'fig2_per_language.{ext}',dpi=180)
