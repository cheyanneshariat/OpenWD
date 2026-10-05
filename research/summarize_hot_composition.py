#!/usr/bin/env python3
"""Persist the nine-element screen, including convergence and FITS checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from astropy.io import fits


def main():
    root=Path('results/hot-daz');out=root/'composition-summary';out.mkdir(exist_ok=True)
    labels={'CSi':'C + Si','Light':'+ N/O/Al/P/S (NLTE)','All':'+ Fe/Ni (LTE)'}
    directories={'CSi':root/'composition-csi','Light':root/'composition-light','All':root/'composition-all'}
    new=json.loads((out/'metrics.json').read_text())
    cases={};arrays={}
    for label,path in directories.items():
        meta=json.loads((path/'metadata.json').read_text())
        report=json.loads((path/'comparison/comparison.json').read_text())
        with fits.open(path/'comparison/prediction.fits',checksum=True) as hdus:
            for hdu in hdus:
                if hdu.verify_checksum()!=1 or hdu.verify_datasum()!=1:raise ValueError('invalid FITS checksum')
            if hdus[0].header['METCONV']!=meta['converged']:raise ValueError('inconsistent convergence')
            if hdus[0].header['LTEELEM']!=','.join(meta['lte_background_elements']):raise ValueError('inconsistent LTE elements')
            for e,ab in meta['abundances'].items():
                if not np.isclose(hdus[0].header[e.upper()+'_H'],10**ab,rtol=1e-13,atol=0):
                    raise ValueError('FITS abundance does not roundtrip')
        cases[label]=dict(directory=str(path),metadata=meta,
            progress=json.loads((path/'progress.json').read_text()),original_diagnostic_metrics=report['metrics'],
            file_sha256={name:hashlib.sha256((path/name).read_bytes()).hexdigest()
                         for name in ['spectrum.npz','populations.npz','metadata.json','comparison/prediction.fits']})
        with np.load(path/'comparison/comparison_arrays.npz') as p:arrays[label]={k:p[k].copy() for k in p.files}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    from compare_hot_daz_benchmark import REGIONS
    colors=['#7c90a0','#d59324','#c3473d']
    for ax,(region,config) in zip(axes.flat,REGIONS.items()):
        base=arrays['CSi'];w=base[region+'_wavelength'];obs=base[region+'_observed']
        ax.plot(w,obs,c='.25',lw=.7,label='Observed STIS')
        for (label,array),color in zip(arrays.items(),colors):
            ax.plot(w,array[region+'_model'],c=color,lw=1.1,label=labels[label])
        ax.set(title=config['label'],xlabel='Observed vacuum wavelength (Å)',ylabel='Normalized flux')
        if region!='lyalpha':ax.set_ylim(max(0,np.min(obs)-.05),1.1)
        ax.axhline(1.,color='.8',lw=.5)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('G191-B2B: effect of the additional reported elements\nFixed host and published abundances; 8-iteration screens, unconverged')
    fig.savefig(out/'carbon-silicon.png',dpi=160);fig.savefig(out/'carbon-silicon.pdf');plt.close(fig)
    summary=dict(cases=cases,additional_diagnostics=new,
        atomic_audit=json.loads((root/'composition-all/atomic-audit.json').read_text()),
        input_manifest=json.loads((root/'multimetal-data/manifest.json').read_text()),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path('src/wd_spectra/hot_trace_metals.py'),Path('research/hot_trace_composition.py'),
            Path('research/explore_hot_composition.py'),Path('research/compare_hot_composition.py'),
            Path('research/compare_hot_daz_benchmark.py'),Path(__file__)]},
        verification='147 distinct targeted tests passed; added atoms recover Planck LTE; Fe/Ni Kirchhoff; FITS checksums/abundances checked',
        scope='Nine Table-10 elements; seven NLTE, Fe/Ni LTE. Fixed H/He structure. No fitted abundances or levitation profile.',
        interpretation='C III aperture EWs improve, but Si IV 1402 worsens and O IV/Fe V/Ni V remain weak. Population defects remain near unity; spectral trends are exploratory.')
    text=json.dumps(summary,indent=2)+'\n'
    (out/'summary.json').write_text(text)
    Path('docs/development/history/hot-daz-g191-b2b-composition.json').write_text(text)
    print('Saved composition comparison, audit, and checksum-verified history.')

if __name__=='__main__':main()
