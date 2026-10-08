"""Plot only after all timing processes finish."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
p=json.loads((D/'manifest.json').read_text());assert p['status']=='completed'
v=json.loads((D/'summary.json').read_text());rows=v['rows'];x=np.arange(len(rows))
fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
for ax,key,title,color in zip(axes[:2],['offload_vs_main_pct','offload_vs_folded_gpu_pct'],['PR offload vs upstream main GPU','PR offload vs PR folded GPU'],['#3978ad','#8a54a2']):
 for i,r in enumerate(rows):
  if key not in r:ax.text(i,.04,'OOM',transform=ax.get_xaxis_transform(),rotation=90,ha='center',fontsize=8);continue
  a=r[key];ax.errorbar(i,a['mean'],yerr=[[a['mean']-a['lower_95']],[a['upper_95']-a['mean']]],fmt='o',color=color,capsize=3)
 ax.axhline(0,color='gray',linestyle='--',linewidth=.8);ax.set_ylabel('Latency change (%)');ax.set_title(title);ax.grid(axis='y',alpha=.25)
for i,r in enumerate(rows):
 a,b=r['placements']['main_gpu'],r['placements']['head_offload']
 if a['status']==b['status']=='completed':axes[2].bar(i,a['gpu_peak_allocated_gib']-b['gpu_peak_allocated_gib'],color='#35806c')
 else:axes[2].text(i,.04,'OOM',transform=axes[2].get_xaxis_transform(),rotation=90,ha='center',fontsize=8)
axes[2].set_ylabel('GPU memory saved (GiB)');axes[2].set_title('Peak allocated memory: main GPU minus PR offload');axes[2].grid(axis='y',alpha=.25)
axes[2].set_xticks(x,[f"{r['mode']}\nb{r['batch']}/L{r['length']}" for r in rows],rotation=60,ha='right',fontsize=8)
fig.suptitle('Fresh upstream / PR-head paired measurements',fontsize=14)
fig.text(.01,.01,'RTX 5090 | 2.8365B BF16 | seeded random weights | last-token logits + KV | 3 process blocks; paired 95% t intervals (df=2).\nNegative latency change is faster. Main includes per-token normalization; folded GPU controls for folding. Generation: prefix +128 steps. OOM is not zero.',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.96))
for ext in ('png','pdf','svg'):fig.savefig(D/f'comparison.{ext}',dpi=180)
