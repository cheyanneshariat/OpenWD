"""Plot the saved DAB grid and one-parameter spectrum sequences.

This script reads released spectra. It does not run OpenWD or interpolate models.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'COMPLETE_NUMERICAL':'#237a57','NUMERICAL_NONCONVERGENCE':'#ba3838',
        'DOMAIN_OR_UNSUPPORTED_CONFIG':'#8c64ad','TIMEOUT':'#cc8b25','UNATTEMPTED':'#bcbcbc'}
MARKERS={'COMPLETE_NUMERICAL':'o','NUMERICAL_NONCONVERGENCE':'x','DOMAIN_OR_UNSUPPORTED_CONFIG':'^','TIMEOUT':'s','UNATTEMPTED':'.'}
LABELS={'COMPLETE_NUMERICAL':'Numerical checks passed','NUMERICAL_NONCONVERGENCE':'Numerical check failed',
        'DOMAIN_OR_UNSUPPORTED_CONFIG':'Unsupported','TIMEOUT':'Time limit','UNATTEMPTED':'Not started'}
LINESTYLES=['-',(0,(5,2)),(0,(2,1)),(0,(5,1,1,1)),(0,(1,1)),(0,(8,2,2,2)),(0,(3,1,1,1,1,1)),(0,(9,2))]

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def coverage(rows,out):
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    statuses=set(r['status'] for r in rows)
    for ax,h in zip(axes.flat,[-6,-4,-2,0,2,4]):
        subset=[r for r in rows if float(r['log_H_He'])==h];counts=Counter(r['status'] for r in subset)
        for status in COLORS:
            q=[r for r in subset if r['status']==status]
            if q:ax.scatter([float(r['teff_K'])/1000 for r in q],[float(r['logg']) for r in q],
                            c=COLORS[status],marker=MARKERS[status],s=30,zorder=2 if status=='COMPLETE_NUMERICAL' else 3)
        ax.set_title(r'$\log_{10}[N(\mathrm{H})/N(\mathrm{He})]='+f'{h:g}'+r'$'+f'  ({counts["COMPLETE_NUMERICAL"]}/{len(subset)})',fontsize=12)
        ax.set_xlim(10.8,41.5);ax.set_ylim(9.7,5.8)
    for ax in axes[-1]:ax.set_xlabel(r'$T_{\mathrm{eff}}$ (kK)')
    for ax in axes[:,0]:ax.set_ylabel(r'$\log g$ (cgs)')
    fig.legend([Line2D([],[],marker=MARKERS[s],color=COLORS[s],ls='') for s in COLORS if s in statuses],
               [LABELS[s] for s in COLORS if s in statuses],loc='lower center',ncol=3,fontsize=11,frameon=False)
    completed=sum(r['status']=='COMPLETE_NUMERICAL' for r in rows)
    fig.suptitle(f'DAB/DBA production grid: {completed}/{len(rows)} numerical completions',fontsize=15)
    fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.89,hspace=.3,wspace=.1)
    for ext in ['png','pdf']:fig.savefig(out/f'dab_coverage.{ext}',dpi=250)
    plt.close(fig)

def sequence(bank,rows,out,name,axis,values,fixed,title):
    selected=[]
    for value in values:
        config=dict(fixed);config[axis]=value
        matches=[r for r in rows if all(float(r[k])==float(v) for k,v in config.items())]
        assert len(matches)==1 and matches[0]['status']=='COMPLETE_NUMERICAL',config
        r=matches[0];p=bank/r['spectrum_path'];assert file_sha(p)==r['spectrum_sha256']
        spectrum=np.loadtxt(p);assert np.isfinite(spectrum).all() and np.all(spectrum[:,1]>0)
        selected.append((r,spectrum))
    wave=selected[0][1][:,0]
    assert all(np.array_equal(s[:,0],wave) for r,s in selected),'Cannot form a native-sample ratio on different grids'
    baseline={'log_H_He':-2,'teff_K':20000,'logg':8}[axis]
    reference=next(s[:,1] for r,s in selected if float(r[axis])==baseline)
    optical=(wave>=4000)&(wave<=7000)
    fig,axes=plt.subplots(2,2,figsize=(12,7.2),gridspec_kw={'height_ratios':[2.1,1],'hspace':.08,'wspace':.28},sharex='col')
    palette=plt.colormaps['viridis'](np.linspace(.05,.9,len(selected)))
    for i,((r,s),color) in enumerate(zip(selected,palette)):
        flux=s[:,1];value=float(r[axis]);style=LINESTYLES[i]
        label=(rf'$\log_{{10}}[N(\mathrm{{H}})/N(\mathrm{{He}})]={value:g}$' if axis=='log_H_He' else
               rf'$T_{{\mathrm{{eff}}}}={value/1000:g}\,\mathrm{{kK}}$' if axis=='teff_K' else rf'$\log g={value:g}$')
        axes[0,0].plot(wave,wave*flux,color=color,ls=style,lw=1.6,label=label)
        axes[0,1].plot(wave[optical],flux[optical]/1e8,color=color,ls=style,lw=1.6)
        axes[1,0].plot(wave,flux/reference,color=color,ls=style,lw=1.3)
        axes[1,1].plot(wave[optical],(flux/reference)[optical],color=color,ls=style,lw=1.3)
    axes[0,0].set_yscale('log')
    for ax in axes[:,0]:ax.set_xscale('log');ax.set_xlim(900,30000)
    for ax in axes[:,1]:ax.set_xlim(4000,7000)
    axes[0,1].set_ylim(bottom=0)
    axes[0,0].set_ylabel(r'$\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)')
    axes[0,1].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
    for ax in axes[1]:
        ax.axhline(1,color='.55',lw=.7,zorder=0)
        ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$F_\lambda/F_{\lambda,\mathrm{ref}}$')
    # Optical ratios have a linear scale; a log scale retains the full broad-band range.
    axes[1,0].set_yscale('log')
    axes[0,0].set_title('Native surface spectrum',fontsize=12)
    axes[0,1].set_title('Optical hydrogen and helium features',fontsize=12)
    reference_label=(r'$\log_{10}[N(\mathrm{H})/N(\mathrm{He})]=-2$' if axis=='log_H_He' else
                     r'$T_{\mathrm{eff}}=20\,\mathrm{kK}$' if axis=='teff_K' else r'$\log g=8$')
    fig.suptitle(title,fontsize=15,y=.98)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4 if len(values)>6 else 3,fontsize=10,frameon=False)
    fig.text(.5,.003,'Ratios use '+reference_label+'; saved fluxes are unchanged.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.085,right=.985,bottom=.23,top=.87)
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=250)
    plt.close(fig)
    return dict(variable=axis,fixed_parameters=fixed,reference_value=baseline,wavelength_interpolation='none',
        flux_normalization='none',ratio_definition='Saved native F_lambda divided by the named reference spectrum at identical samples',
        models=[dict(model_id=r['model_id'],parameters={k:float(r[k]) for k in ['teff_K','logg','log_H_He']},spectrum_sha256=r['spectrum_sha256']) for r,s in selected])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--coverage-table',type=Path,
                    help='Optional selected status table; spectrum sequences still use the unchanged original bank')
    ap.add_argument('--style',type=Path,default=Path(__file__).with_name('grid-style.mplstyle'));args=ap.parse_args()
    rows=list(csv.DictReader((args.bank/'requests.csv').open()));assert len(rows)==len({r['model_id'] for r in rows})==336
    assert Counter(r['status'] for r in rows)==Counter(COMPLETE_NUMERICAL=326,DOMAIN_OR_UNSUPPORTED_CONFIG=9,NUMERICAL_NONCONVERGENCE=1)
    args.output.mkdir(parents=True,exist_ok=False);plt.style.use(args.style)
    coverage_rows=list(csv.DictReader(args.coverage_table.open())) if args.coverage_table else rows
    expected={(float(r['teff_K']),float(r['logg']),float(r['log_H_He'])) for r in rows}
    assert len(coverage_rows)==336 and len({r['model_id'] for r in coverage_rows})==336
    assert {(float(r['teff_K']),float(r['logg']),float(r['log_H_He'])) for r in coverage_rows}==expected
    assert all(r['status'] in COLORS for r in coverage_rows)
    coverage(coverage_rows,args.output)
    records={}
    records['dab_abundance_sequence']=sequence(args.bank,rows,args.output,'dab_abundance_sequence','log_H_He',[-6,-4,-2,0,2,4],
        dict(teff_K=20000,logg=8),r'Change H/He abundance: $T_{\mathrm{eff}}=20\,\mathrm{kK}$, $\log g=8$')
    records['dab_temperature_sequence']=sequence(args.bank,rows,args.output,'dab_temperature_sequence','teff_K',[12000,15000,20000,25000,30000,35000,40000],
        dict(logg=8,log_H_He=-2),r'Change temperature: $\log g=8$, $\log_{10}[N(\mathrm{H})/N(\mathrm{He})]=-2$')
    records['dab_gravity_sequence']=sequence(args.bank,rows,args.output,'dab_gravity_sequence','logg',[6,6.5,7,7.5,8,8.5,9,9.5],
        dict(teff_K=20000,log_H_He=-2),r'Change gravity: $T_{\mathrm{eff}}=20\,\mathrm{kK}$, $\log_{10}[N(\mathrm{H})/N(\mathrm{He})]=-2$')
    record=dict(index_sha256=file_sha(args.bank/'requests.csv'),source_commit=json.loads((args.bank/'dataset.json').read_text())['source_commit'],
        scientific_identity=json.loads((args.bank/'dataset.json').read_text())['scientific_identity'],plots=records,
        coverage_index_sha256=file_sha(args.coverage_table if args.coverage_table else args.bank/'requests.csv'),
        coverage_status_counts=dict(Counter(r['status'] for r in coverage_rows)),
        interpretation='Parameter response of saved numerically accepted models; no external physical/interpolation/recovery validation')
    (args.output/'dab_plot_provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(plots=4,unique_spectra=len({m['model_id'] for r in records.values() for m in r['models']}),source_commit=record['source_commit'])))

if __name__=='__main__':main()
