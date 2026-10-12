"""Plot two verified, saved hot spectra; no atmosphere calculation."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--verification',type=Path,default=Path(__file__).with_name('hot-extension-2026-10-12')/'ARCHIVE_VERIFICATION.json')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();receipt=json.loads(a.verification.read_text())
    assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==receipt['archive_sha256']
    with tarfile.open(a.archive) as bank:
        for n,h in receipt['member_hashes'].items():assert hashlib.sha256(bank.extractfile(n).read()).hexdigest()==h
        rows=json.loads(bank.extractfile('native-models.json').read());assert len(rows)==2
        spectra={}
        for r in rows:
            assert r['status']=='COMPLETE_NUMERICAL' and r['audit']['certificate']['verified'] and r['audit']['independent_final_spectrum_source_verified']
            raw=bank.extractfile(r['spectrum_archive_path']).read();assert hashlib.sha256(raw).hexdigest()==r['spectrum_sha256']
            v=np.loadtxt(io.BytesIO(raw));assert np.isfinite(v).all() and (np.diff(v[:,0])>0).all() and (v[:,1]>=0).all()
            spectra[r['family']]=v
    assert set(spectra)=={'DO','DAO'}
    a.output.mkdir(parents=True,exist_ok=False)
    plt.style.use(Path(__file__).with_name('grid-style.mplstyle'))
    plt.rcParams.update({'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    titles=[r'DO, 60,000 K, $\log g=9$',r'DAO, 60,000 K, $\log g=8$']
    for ax,fam,title,color in zip(axes,['DO','DAO'],titles,['#24496a','#b56738']):
        v=spectra[fam];m=(v[:,0]>=3500)&(v[:,0]<=7000);ax.plot(v[m,0],v[m,1]/1e8,color=color,lw=1)
        ax.set_title(title,fontsize=15);ax.set_xlim(3500,7000);ax.set_ylim(bottom=0)
        ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)');ax.set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.89,wspace=.28)
    assert not fig.texts and not fig.legends and all(not ax.texts and ax.get_legend() is None for ax in axes)
    for ext in ['png','pdf']:fig.savefig(a.output/f'hot_extension.{ext}',dpi=250)
    plt.close(fig)
    (a.output/'PROVENANCE.json').write_text(json.dumps(dict(archive_sha256=receipt['archive_sha256'],flux_normalization='none',wavelength_resampling='none',spectrum_hashes={r['family']:r['spectrum_sha256'] for r in rows}),indent=2)+'\n')
if __name__=='__main__':main()
