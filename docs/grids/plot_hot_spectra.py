"""Plot checksummed saved DO/DAO products; no model calculation is performed."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--records',type=Path,default=Path(__file__).with_name('hot-2026-10-12')/'native-models.json')
    parser.add_argument('--verification',type=Path,default=Path(__file__).with_name('hot-2026-10-12')/'ARCHIVE_VERIFICATION.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=json.loads(args.verification.read_text())
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest()==receipt['archive_sha256']
    rows=json.loads(args.records.read_text());assert len(rows)==8
    spectra=[]
    with tarfile.open(args.archive) as archive:
        names={m.name for m in archive.getmembers() if m.isfile()}
        assert names==set(receipt['member_hashes'])
        for name,digest in receipt['member_hashes'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
        assert json.loads(archive.extractfile('native-models.json').read())==rows
        for row in rows:
            if row['status']!='COMPLETE_NUMERICAL':continue
            cert=row['audit']['certificate'];assert cert['verified'] and not cert['failures']
            assert all(cert['checks'][name]['passed'] for name in cert['required_checks'])
            assert row['audit']['independent_final_spectrum_source_verified']
            raw=archive.extractfile(row['spectrum_archive_path']).read()
            assert hashlib.sha256(raw).hexdigest()==row['spectrum_sha256']
            values=np.loadtxt(io.BytesIO(raw))
            assert np.isfinite(values).all() and (np.diff(values[:,0])>0).all() and (values[:,1]>=0).all()
            spectra.append((row,values))
    assert len(spectra)==4
    args.output.mkdir(parents=True,exist_ok=False)
    plt.style.use(Path(__file__).with_name('grid-style.mplstyle'))
    plt.rcParams.update({'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    spectra.sort(key=lambda x:(x[0]['family'],x[0]['config']['logg']))
    for ax,(row,values) in zip(axes.flat,spectra):
        c=row['config'];mask=(values[:,0]>=3500)&(values[:,0]<=7000)
        ax.plot(values[mask,0],values[mask,1]/1e8,color='#24496a',lw=.9)
        title=f"{row['family']}  {c['effective_temperature']/1000:g} kK, log g {c['logg']:g}"
        if row['family']=='DAO':title+=f", log(H/He) {c['log_hydrogen_to_helium']:+g}"
        ax.set_title(title,fontsize=13)
        ax.set_xlim(3500,7000);ax.set_ylim(bottom=0)
        ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
    fig.subplots_adjust(left=.085,right=.98,bottom=.085,top=.94,wspace=.28,hspace=.42)
    save(fig,args.output/'hot_spectra')
    fig,axes=plt.subplots(1,2,figsize=(9,3.9))
    for ax,family in zip(axes,['DO','DAO']):
        subset=[r for r in rows if r['family']==family]
        for row in subset:
            c=row['config'];accepted=row['status']=='COMPLETE_NUMERICAL'
            if accepted:ax.scatter(c['effective_temperature']/1000,c['logg'],s=54,color='#237a57',marker='o')
            else:
                assert row['status']=='TIMEOUT'
                ax.scatter(c['effective_temperature']/1000,c['logg'],s=83,facecolors='none',edgecolors='#777777',marker='s',linewidths=1.4)
        ax.set_title(f'{family}  (2/4)',fontsize=15)
        ax.set_xlim(30,130);ax.set_ylim(6.8,8.2);ax.set_yticks([7,7.5,8])
        ax.set_xlabel(r'$T_{\rm eff}$ (kK)');ax.set_ylabel(r'$\log g$ (cgs)')
    fig.subplots_adjust(left=.085,right=.97,bottom=.17,top=.89,wspace=.3)
    save(fig,args.output/'hot_outcomes')
    (args.output/'plot_provenance.json').write_text(json.dumps(dict(archive_sha256=receipt['archive_sha256'],accepted_spectra=4,timeouts=4,flux_normalization='none',wavelength_interpolation='none',display_band_angstrom=[3500,7000],physical_accuracy_verified=False,independent_depth_convergence_verified=False),indent=2)+'\n')


def save(fig,path):
    assert not fig.legends and not fig.texts
    assert all(not ax.texts and ax.get_legend() is None for ax in fig.axes)
    for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=250)
    plt.close(fig)


if __name__=='__main__':main()
