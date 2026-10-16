"""Plot measured fixed-atmosphere line-policy changes, with native arrays."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
d=json.loads((R/'COMPARISON.json').read_text());assert d['status']=='COMPLETE'
plt.style.use(str(R/'plot-style.mplstyle'))
fig,axes=plt.subplots(4,2,figsize=(11,10),layout='constrained')
names={'dz-pg1225':'PG 1225−079','dz-j0738':'J0738+1835','daz-g149_28':'G149−28','daz-galex1931':'GALEX J1931+0117'}
for row,(a,b) in zip(d['records'],axes):
    path=R/(row['case']+'.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==row['output_sha256']
    with np.load(path,allow_pickle=False) as z:
        w=z['wavelength'];old=z['previous_F_lambda'];new=z['current_F_lambda']
    lo,hi=(1150,3000) if row['family']=='DZ' else (max(3500,float(w[0])),min(7000,float(w[-1])))
    take=(w>=lo)&(w<=hi)
    a.plot(w[take],old[take]/1e8,color='#c57535',lw=1,label='Previous line defaults')
    a.plot(w[take],new[take]/1e8,color='#2b628c',lw=.9,label='Latest line defaults')
    a.set_title(names[row['case']],fontsize=13,loc='left');a.set_ylim(bottom=0)
    b.axhline(0,color='.6',lw=.8)
    b.plot(w[take],100*(new[take]/old[take]-1),color='#2b628c',lw=.8)
    for ax in [a,b]:ax.set_xlim(lo,hi)
    a.set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)',fontsize=12)
    b.set_ylabel('Flux change (%)',fontsize=12)
    for ax in [a,b]:ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)',fontsize=12);ax.tick_params(labelsize=11)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=12)
for ext in ['png','pdf']:fig.savefig(R/f'line_policy_changes.{ext}',dpi=250)
plt.close(fig)
(R/'CAPTION.md').write_text('Both spectra use the same saved atmosphere, composition and native wavelength samples. Orange uses the previous Kurucz line defaults; blue uses latest main (00503a0). Left panels show absolute surface flux; right panels show the fractional flux change. DZ panels show the ultraviolet; DAZ panels show the optical range available in these controls. All changes remain visible, including large fractional changes in weak UV flux. No spectrum was normalized or resampled. These fixed-atmosphere comparisons isolate the line policy. They do not test a cold atmosphere, depth convergence or agreement with observations.\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'METAL_FIGURE_INPUTS.json').write_text(json.dumps(dict(comparison_sha256=sha(R/'COMPARISON.json'),cases=4,
    normalization='none',wavelength_resampling='none',files={p.name:sha(p) for p in [R/'line_policy_changes.png',R/'line_policy_changes.pdf']}),indent=2)+'\n')
