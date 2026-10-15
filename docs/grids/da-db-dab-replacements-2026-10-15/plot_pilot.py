"""Plot verified pilot old/new absolute spectra and differences; no model solver."""
from pathlib import Path
import argparse,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--style',type=Path,required=True);a=p.parse_args()
 c=json.loads((a.inputs/'COMPARISON.json').read_text());rows=[max([r for r in c['records'] if r.get('comparison_status')=='EXACT_NATIVE_COMPARISON' and r['family']==p],key=lambda r:r['bands']['UV']['integrated_absolute_relative_change']) for p in ['DA','DB','DAB']];assert len(rows)==3
 arrays=[]

 for row in rows:
  f=a.inputs/row['comparison_file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['comparison_sha256']
  with np.load(f,allow_pickle=False) as z:values=[z[k] for k in ['wavelength','old_surface_flux_lambda','new_surface_flux_lambda']]
  wave,old,new=values;assert np.isfinite(values).all() and (np.diff(wave)>0).all() and (old>0).all()
  # Recheck stored integrated differences from the actual shared arrays.
  for b in row['bands'].values():
   m=(wave>=b['lo_angstrom'])&(wave<=b['hi_angstrom']);computed=np.trapz(np.abs(new[m]-old[m]),wave[m])/np.trapz(old[m],wave[m]);assert computed==b['integrated_absolute_relative_change']
  arrays.append(values)
 a.output.mkdir(parents=True,exist_ok=True);plt.style.use(a.style);plt.rcParams.update({'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})
 for name,lo,hi in [('UV',1150,3000),('optical',3000,10000)]:
  fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
  for axpair,row,values in zip(axes,rows,arrays):
   wave,old,new=values;m=(wave>=lo)&(wave<=hi);title=row['family']+', '+str(int(row['config']['effective_temperature']))+r' K, $\log g='+str(row['config']['logg'])+'$'
   if row['family']=='DAB':title+=r', $\log[N(H)/N(He)]='+str(row['config']['log_hydrogen_to_helium'])+'$'
   axpair[0].plot(wave[m],old[m]/1e8,'--',color='black',lw=1.1);axpair[0].plot(wave[m],new[m]/1e8,'-',color='#24496a',lw=1.0)
   axpair[1].plot(wave[m],100*(new[m]-old[m])/old[m],color='#24496a',lw=.9);axpair[1].axhline(0,color='.6',lw=.6)
   for ax in axpair:ax.set_title(title,fontsize=12);ax.set_xlim(lo,hi);ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
   axpair[0].set_ylim(bottom=0);axpair[0].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)');axpair[1].set_ylabel(r'$(F_{\rm new}-F_{\rm old})/F_{\rm old}$ (%)')
  assert not fig.texts and not fig.legends and all(not ax.texts and ax.get_legend() is None for ax in axes.flat)
  for ext in ['png','pdf']:fig.savefig(a.output/('pilot_'+name+'.'+ext),dpi=250)
  plt.close(fig)
 (a.output/'plot_provenance.json').write_text(json.dumps(dict(source_commit=c['source_commit'],old_source_commits={r['case_id']:r['old_source_commit'] for r in rows},cases=[x['case_id'] for x in rows],comparison_sha256=hashlib.sha256((a.inputs/'COMPARISON.json').read_bytes()).hexdigest(),flux_normalization='none; physical axis unit scaling only',wavelength_resampling='none',metrics_rechecked=9,selection='Largest integrated absolute UV change among matched pilot pairs in each family (DA2,DB3,DAB5); six accepted models without accepted predecessors are separate. These are diagnostic source-change comparisons, not observational fits',plot_text='titles, axes and ticks only',physical_accuracy_verified=False,independent_depth_convergence_verified=False),indent=2)+'\n')
if __name__=='__main__':main()
