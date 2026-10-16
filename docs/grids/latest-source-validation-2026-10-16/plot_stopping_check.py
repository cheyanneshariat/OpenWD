from pathlib import Path
import json,hashlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;d=json.loads((R/'DIAGNOSIS.json').read_text())
plt.style.use(str(R/'plot-style.mplstyle'))
fig,ax=plt.subplots(figsize=(6.7,5.2),layout='constrained')
counts={}
for fam,color,marker in [('DA','#2b628c','o'),('DB','#c57535','^')]:
    rows=[x for x in d['records'] if x['family']==fam and x['stationarity_measured'] and x['accepted_step'] is not None and x['accepted_step']>0 and x['measured_unrestricted']>0]
    counts[fam]=len(rows)
    ax.scatter([x['accepted_step']/x['temperature_tolerance'] for x in rows],
        [x['measured_unrestricted']/x['temperature_tolerance'] for x in rows],
        c=color,marker=marker,s=34,label=f'{fam} ({len(rows)})',alpha=.8)
ax.axvline(1,color='.35',ls='--',lw=1);ax.axhline(1,color='.35',ls='--',lw=1)
ax.set(xscale='log',yscale='log',xlabel='Accepted temperature step / limit',ylabel='Recorded unrestricted correction / limit')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,fontsize=12,frameon=False)
for ext in ['png','pdf']:fig.savefig(R/f'stopping_check.{ext}',dpi=250)
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'STOPPING_FIGURE_INPUTS.json').write_text(json.dumps(dict(input_sha256=sha(R/'DIAGNOSIS.json'),plotted=counts,
    mismatch_cases=d['measured_step_mismatches'],files={p.name:sha(p) for p in [R/'stopping_check.png',R/'stopping_check.pdf']}),indent=2)+'\n')
