"""Reproduce DZ old/new absolute spectra and differences; no model solver."""
from pathlib import Path
import argparse,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--style',type=Path,required=True);a=p.parse_args()
 c=json.loads((a.inputs/'COMPARISON.json').read_text());rows=sorted(c['records'],key=lambda r:r['config']['effective_temperature']);assert len(rows)==4 and all(r['comparison_status']=='EXACT_NATIVE_COMPARISON' for r in rows)
 arrays=[]

 for row in rows:
  f=a.inputs/row['comparison_file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['comparison_sha256']
  with np.load(f,allow_pickle=False) as z:values=[z[k] for k in ['wavelength','old_surface_flux_lambda','new_surface_flux_lambda']]
  wave,old,new=values;assert np.isfinite(values).all() and (np.diff(wave)>0).all() and (old>0).all()
  # Recheck stored integrated differences from the actual shared arrays.
  for b in row['bands'].values():
   m=(wave>=b['lo_angstrom'])&(wave<=b['hi_angstrom']);computed=np.trapz(np.abs(new[m]-old[m]),wave[m])/np.trapz(old[m],wave[m]);assert computed==b['integrated_absolute_relative_change']
  arrays.append(values)
 a.output.mkdir(parents=True,exist_ok=True);plt.style.use(a.style);plt.rcParams.update({'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10})
 for name,lo,hi in [('UV',1150,3000),('optical',3500,7000)]:
  fig,axes=plt.subplots(4,2,figsize=(11,12),layout=None)
  for axpair,row,values in zip(axes,rows,arrays):
   wave,old,new=values;m=(wave>=lo)&(wave<=hi);title=row['pattern']+', '+str(int(row['config']['effective_temperature']))+r' K, $\log g='+str(row['config']['logg'])+'$'
   axpair[0].plot(wave[m],old[m]/1e8,'--',color='black',lw=1.1);axpair[0].plot(wave[m],new[m]/1e8,'-',color='#24496a',lw=1.0)
   axpair[1].plot(wave[m],100*(new[m]-old[m])/old[m],color='#24496a',lw=.9);axpair[1].axhline(0,color='.6',lw=.6)
   for ax in axpair:ax.set_title(title,fontsize=12);ax.set_xlim(lo,hi);ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
   axpair[0].set_ylim(bottom=0);axpair[0].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)');axpair[1].set_ylabel(r'$(F_{\rm new}-F_{\rm old})/F_{\rm old}$ (%)')
  fig.subplots_adjust(left=.095,right=.965,bottom=.055,top=.965,hspace=.57,wspace=.33)
  assert not fig.texts and not fig.legends and all(not ax.texts and ax.get_legend() is None for ax in axes.flat)
  fig.canvas.draw()
  renderer=fig.canvas.get_renderer()
  for ax in axes.flat:
   for t in [ax.title,ax.xaxis.label,ax.yaxis.label]+[t for x,t in zip(ax.get_xticks(),ax.get_xticklabels()) if min(ax.get_xlim())<=x<=max(ax.get_xlim())]+[t for y,t in zip(ax.get_yticks(),ax.get_yticklabels()) if min(ax.get_ylim())<=y<=max(ax.get_ylim())]:
    if t.get_visible() and t.get_text():
     b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text(),b)
  for ext in ['png','pdf']:fig.savefig(a.output/('dz_'+name+'.'+ext),dpi=250)
  plt.close(fig)
 (a.output/'plot_provenance.json').write_text(json.dumps(dict(source_commit=c['source_commit'],old_source_commit='cfe2d2ff99a434cd502696c1eb7b2f5daf8d11c5',cases=[x['case_id'] for x in rows],comparison_sha256=hashlib.sha256((a.inputs/'COMPARISON.json').read_bytes()).hexdigest(),flux_normalization='none; physical axis unit scaling only',wavelength_resampling='none',metrics_rechecked=12,selection='All four previously deferred GD40 requests, ordered by temperature; these are diagnostic comparisons, not observational fits',plot_text='titles, axes and ticks only',physical_accuracy_verified=False,independent_depth_convergence_verified=False),indent=2)+'\n')
if __name__=='__main__':main()
