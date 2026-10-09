"""Reproduce the saved cold DB comparison; no atmosphere calculations."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path(__file__).parent/'db-cold-2026-10-09/comparison-input.npz')
    ap.add_argument('--provenance',type=Path,default=Path(__file__).parent/'db-cold-2026-10-09/comparison-input.json')
    ap.add_argument('--style',type=Path,default=Path(__file__).with_name('grid-style.mplstyle'))
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.provenance.read_text());assert hashlib.sha256(a.data.read_bytes()).hexdigest()==p['data_sha256']
    a.output.mkdir(parents=True,exist_ok=False);plt.style.use(a.style)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':11,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10})
    fig,axes=plt.subplots(6,2,figsize=(11,14.2),gridspec_kw={'height_ratios':[2,1]*3},layout='constrained')
    with np.load(a.data,allow_pickle=False) as inputs:
        for row,g in enumerate([7,7.5,8]):
            for col,band in enumerate(['broad','optical']):
                wave=inputs[f'g{g:g}_{band}_wave'];new=inputs[f'g{g:g}_{band}_openwd'];ref=inputs[f'g{g:g}_{band}_montreal']
                assert np.isfinite(new).all() and np.isfinite(ref).all() and np.all(ref>0)
                top,bottom=axes[2*row,col],axes[2*row+1,col]
                top.plot(wave,ref,color='#275e83',lw=1.4,label='Montreal')
                top.plot(wave,new,color='#e76f00',lw=1.3,ls='--',label='openWD')
                top.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')
                if col==0:
                    top.set_xscale('log');top.set_yscale('log');bottom.set_xscale('log');bottom.set_yscale('symlog',linthresh=10)
                    top.text(.97,.07,rf'$T_{{\rm eff}}=5,000$ K, $\log g={g:g}$',transform=top.transAxes,ha='right',va='bottom',fontsize=11)
                else:
                    top.ticklabel_format(axis='y',style='sci',scilimits=(0,0));top.legend(loc='lower right',fontsize=9,frameon=False)
                bottom.plot(wave,100*(new/ref-1),color='#e76f00',lw=1)
                bottom.axhline(0,color='.4',ls=':',lw=.8)
                bottom.set_ylabel('Difference [%]\n(symlog)' if col==0 else 'Difference [%]')
                for ax in [top,bottom]:ax.set_xlim(wave[0],wave[-1])
                bottom.set_xlabel(r'Vacuum wavelength [$\AA$]')
    # No overall or panel title. Parameter annotations and caption retain identity.
    for ext in ['png','pdf']:fig.savefig(a.output/f'db_montreal_spectra.{ext}',dpi=250)
    plt.close(fig)

if __name__=='__main__':main()
