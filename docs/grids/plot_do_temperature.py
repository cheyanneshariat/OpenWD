"""Reproduce two saved DO spectra from the verified continuation archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--verification',type=Path,default=Path(__file__).with_name('do-continuation-2026-10-12')/'ARCHIVE_VERIFICATION.json')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();receipt=json.loads(args.verification.read_text())
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest()==receipt['archive_sha256']
    spectra=[]
    with tarfile.open(args.archive) as bank:
        for name,h in receipt['member_hashes'].items():assert hashlib.sha256(bank.extractfile(name).read()).hexdigest()==h
        rows=json.loads(bank.extractfile('native-models.json').read());assert len(rows)==2
        for row in rows:
            assert row['family']=='DO' and row['status']=='COMPLETE_NUMERICAL' and row['config']['logg']==8
            assert row['audit']['certificate']['verified'] and row['audit']['independent_final_spectrum_source_verified']
            raw=bank.extractfile(row['spectrum_archive_path']).read();assert hashlib.sha256(raw).hexdigest()==row['spectrum_sha256']
            v=np.loadtxt(io.BytesIO(raw));assert np.isfinite(v).all() and (np.diff(v[:,0])>0).all() and (v[:,1]>=0).all()
            spectra.append((row['config']['effective_temperature'],v))
    spectra.sort(key=lambda x:x[0]);assert [x[0] for x in spectra]==[60000,70000]
    args.output.mkdir(parents=True,exist_ok=False)
    plt.style.use(Path(__file__).with_name('grid-style.mplstyle'))
    plt.rcParams.update({'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for ax,band in zip(axes,[(3500,7000),(4300,4900)]):
        for (_,v),color,ls in zip(spectra,['#24496a','#b56738'],['-','--']):
            mask=(v[:,0]>=band[0])&(v[:,0]<=band[1]);ax.plot(v[mask,0],v[mask,1]/1e8,color=color,ls=ls,lw=1)
        ax.set_title(r'DO, $\log g=8$',fontsize=15)
        ax.set_xlim(*band);ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
    axes[0].set_ylim(bottom=0)
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.89,wspace=.28)
    assert not fig.texts and not fig.legends and all(not ax.texts and ax.get_legend() is None for ax in axes)
    for ext in ['png','pdf']:fig.savefig(args.output/f'do_temperature.{ext}',dpi=250)
    plt.close(fig)
    (args.output/'plot_provenance.json').write_text(json.dumps(dict(archive_sha256=receipt['archive_sha256'],temperatures_K=[60000,70000],logg=8,flux_normalization='none',wavelength_resampling='none',physical_accuracy_verified=False,independent_depth_convergence_verified=False),indent=2)+'\n')

if __name__=='__main__':main()
