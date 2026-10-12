"""Reproduce saved DZ source comparisons; no atmosphere solve or parameter fit."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 manifest=json.loads((a.inputs/'PLOT_INPUT_HASHES.json').read_text())
 for n,h in manifest.items():assert hashlib.sha256((a.inputs/n).read_bytes()).hexdigest()==h
 report=json.loads((a.inputs/'RESULTS.json').read_text());assert len(report['records'])==2
 rows=[]
 for m in report['records']:
  with np.load(a.inputs/(m['id']+'-comparison.npz'),allow_pickle=False) as d:rows.append((m,{k:d[k] for k in d.files}))
 a.output.mkdir(parents=True,exist_ok=False)
 style=a.inputs.parent/'grid-style.mplstyle'
 if not style.exists():style=Path('/Users/bijan1339/.codex/skills/astro-plot-style/assets/shariat-paper.mplstyle')
 plt.style.use(style);plt.rcParams.update({'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})
 for name,bounds in [('fixed_uv',(1250,1320)),('fixed_full_uv',(1150,1500))]:
  fig,axes=plt.subplots(2,2,figsize=(11,6.6),gridspec_kw={'height_ratios':[2,1]},sharex='col')
  for col,(m,d) in enumerate(rows):
   w=d['wavelength'];mask=(w>=bounds[0])&(w<=bounds[1]);old=d['baseline_fixed'];new=d['updated_fixed']
   axes[0,col].plot(w[mask],old[mask]/1e8,color='0.45',ls='--',lw=1)
   axes[0,col].plot(w[mask],new[mask]/1e8,color='#24496a',lw=1)
   axes[0,col].set_title(f"DZ, {m['teff_K']:,.0f} K, $\\log g={m['logg']:.2f}$",fontsize=15)
   axes[0,col].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)');axes[0,col].set_ylim(bottom=0)
   axes[1,col].plot(w[mask],100*(new[mask]/old[mask]-1),color='#24496a',lw=.8);axes[1,col].axhline(0,color='0.65',lw=.6)
   axes[1,col].set_ylabel('Source change (%)');axes[1,col].set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)');axes[1,col].set_xlim(*bounds)
  fig.subplots_adjust(left=.085,right=.98,bottom=.1,top=.92,wspace=.28,hspace=.07)
  assert not fig.texts and not fig.legends and all(not ax.texts and ax.get_legend() is None for ax in axes.flat)
  for ext in ['png','pdf']:fig.savefig(a.output/(name+'.'+ext),dpi=250)
  plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,6.6),gridspec_kw={'height_ratios':[2,1]},sharex='col')
 for col,(m,d) in enumerate(rows):
  x=d['optical_bin_centers'];ref=d['optical_reference_bins'];base=d['optical_baseline_bins'];new=d['optical_updated_bins']
  axes[0,col].plot(x,ref/1e8,color='black',lw=1);axes[0,col].plot(x,base/1e8,color='0.5',ls='--',lw=1);axes[0,col].plot(x,new/1e8,color='#24496a',lw=1)
  axes[0,col].set_title(f"DZ, {m['teff_K']:,.0f} K, $\\log g={m['logg']:.2f}$",fontsize=15);axes[0,col].set_ylim(bottom=0);axes[0,col].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
  axes[1,col].plot(x,100*(base/ref-1),color='0.5',ls='--',lw=1);axes[1,col].plot(x,100*(new/ref-1),color='#24496a',lw=1);axes[1,col].axhline(0,color='0.65',lw=.6);axes[1,col].set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)');axes[1,col].set_ylabel('Model difference (%)');axes[1,col].set_xlim(3000,9000)
 fig.subplots_adjust(left=.085,right=.98,bottom=.1,top=.92,wspace=.28,hspace=.07)
 assert not fig.texts and not fig.legends and all(not ax.texts and ax.get_legend() is None for ax in axes.flat)
 for ext in ['png','pdf']:fig.savefig(a.output/('cold_optical.'+ext),dpi=250)
 plt.close(fig)
if __name__=='__main__':main()
