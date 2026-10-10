"""Plot the saved one-dex DAB/DBA mesh and independent interpolation tests."""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STYLES = {
    'COMPLETE_NUMERICAL':('#237a57','o','Numerical checks passed'),
    'DOMAIN_OR_UNSUPPORTED_CONFIG':('#a5a5a5','^','Unsupported'),
    'TIMEOUT':('#8d8d8d','s','Timed out'),
    'WORKER_CRASH_SIGSEGV':('#555555','x','Worker crash'),
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(fig, output, name):
    for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=250)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--coarse-bank',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).with_name('dab-1dex-2026-10-10'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--style',type=Path,default=Path(__file__).with_name('grid-style.mplstyle'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);plt.style.use(args.style)
    rows=list(csv.DictReader((args.inputs/'selected-requests.csv').open()))
    assert len(rows)==len({r['model_id'] for r in rows})==616
    assert sum(r['status']=='COMPLETE_NUMERICAL' for r in rows)==588
    for line in (args.inputs/'INPUT_SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(args.inputs/name)==expected,name
    fig,axes=plt.subplots(3,4,figsize=(15,9.5),sharex=True,sharey=True)
    for ax,h in zip(axes.flat,range(-6,5)):
        subset=[r for r in rows if float(r['log_H_He'])==h];assert len(subset)==56
        for key,(color,marker,label) in STYLES.items():
            q=[r for r in subset if r['derived_status']==key]
            if q:ax.scatter([float(r['teff_K'])/1000 for r in q],[float(r['logg']) for r in q],color=color,marker=marker,s=26)
        n=sum(r['status']=='COMPLETE_NUMERICAL' for r in subset)
        ax.set_title(r'$\log_{10}[N(\mathrm{H})/N(\mathrm{He})]='+f'{h:g}'+r'$'+f' ({n}/56)',fontsize=11)
        ax.set_xlim(10.8,41.5);ax.set_ylim(9.7,5.8);ax.set_xticks([12,20,30,40]);ax.set_yticks([6,7,8,9])
    axes.flat[-1].set_visible(False)
    for ax in axes[-1,:3]:ax.set_xlabel(r'$T_{\rm eff}$ (kK)',fontsize=14)
    for ax in axes[:,0]:ax.set_ylabel(r'$\log g$ (cgs)',fontsize=14)
    handles=[Line2D([],[],ls='',marker=marker,color=color,label=label) for color,marker,label in STYLES.values()]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.055,right=.985,bottom=.09,top=.95,wspace=.12,hspace=.27)
    save(fig,args.output,'dab_coverage')

    banks=[(args.coarse_bank,list(csv.DictReader((args.coarse_bank/'requests.csv').open()))),
           (args.bank,list(csv.DictReader((args.bank/'requests.csv').open())))]
    selected=[]
    for h in range(-6,5):
        matches=[(bank,r) for bank,records in banks for r in records
                 if float(r['teff_K'])==20000 and float(r['logg'])==8 and float(r['log_H_He'])==h]
        assert len(matches)==1;bank,row=matches[0];assert row['status']=='COMPLETE_NUMERICAL'
        path=bank/row['spectrum_path'];assert sha(path)==row['spectrum_sha256']
        data=np.loadtxt(path);assert np.isfinite(data).all() and (data[:,1]>0).all()
        selected.append((h,row,data))
    wave=selected[0][2][:,0];assert all(np.array_equal(wave,data[:,0]) for _,_,data in selected)
    reference=next(data[:,1] for h,_,data in selected if h==-2);mask=(wave>=4000)&(wave<=7000)
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharex='col',gridspec_kw={'height_ratios':[2.1,1],'hspace':.08,'wspace':.28})
    patterns=['-',(0,(5,2)),(0,(2,1)),(0,(5,1,1,1)),(0,(1,1)),(0,(8,2,2,2)),(0,(3,1,1,1,1,1)),(0,(9,2)),(0,(6,1,2,1)),(0,(4,2,1,2)),(0,(7,1,1,1,1,1))]
    for (h,row,data),color,pattern in zip(selected,plt.colormaps['viridis'](np.linspace(.05,.9,11)),patterns):
        flux=data[:,1];label=rf'$\log_{{10}}[N(\mathrm{{H}})/N(\mathrm{{He}})]={h:g}$'
        axes[0,0].plot(wave,wave*flux,color=color,ls=pattern,lw=1.5,label=label)
        axes[0,1].plot(wave[mask],flux[mask]/1e8,color=color,ls=pattern,lw=1.5)
        axes[1,0].plot(wave,flux/reference,color=color,ls=pattern,lw=1.2)
        axes[1,1].plot(wave[mask],(flux/reference)[mask],color=color,ls=pattern,lw=1.2)
    axes[0,0].set_yscale('log');axes[1,0].set_yscale('log')
    for ax in axes[:,0]:ax.set_xscale('log');ax.set_xlim(900,30000)
    for ax in axes[:,1]:ax.set_xlim(4000,7000)
    axes[0,1].set_ylim(bottom=0)
    axes[0,0].set_ylabel(r'$\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)')
    axes[0,1].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
    for ax in axes[1]:
        ax.axhline(1,color='.55',lw=.7);ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)');ax.set_ylabel(r'$F_\lambda/F_{\lambda,\mathrm{ref}}$')
    fig.suptitle(r'$T_{\rm eff}=20\,\mathrm{kK},\quad \log g=8$',fontsize=15,y=.985)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,fontsize=10,frameon=False)
    fig.text(.5,.003,r'Ratios use $\log_{10}[N(\mathrm{H})/N(\mathrm{He})]=-2$; saved fluxes are unchanged.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.085,right=.985,bottom=.25,top=.89)
    save(fig,args.output,'dab_abundance_sequence')
    report=json.loads((args.inputs/'interpolation/comparison.json').read_text())
    fig,axes=plt.subplots(4,2,figsize=(11,11.7),sharex=True)
    for ax,row in zip(axes.flat,report['records']):
        p=row['parameters'];ax.text(.97,.94,rf'$T_{{\rm eff}}={p["teff_K"]/1000:g}\,\mathrm{{kK}},\ \log g={p["logg"]:g}$'+'\n'+rf'$\log_{{10}}[N(\mathrm{{H}})/N(\mathrm{{He}})]={p["log_H_He"]:g}$',ha='right',va='top',transform=ax.transAxes,fontsize=11,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
        ax.axhline(0,color='.65',lw=.7)
        for y in (-3,3):ax.axhline(y,color='.65',ls=':',lw=.6)
        if row['comparison_status']=='MEASURED':
            with np.load(args.inputs/'interpolation'/f'{row["model_id"]}.npz',allow_pickle=False) as data:
                w=data['wavelength_angstrom'];direct=data['independent_F_lambda'];mask=(w>=3000)&(w<=10000)
                for key,color,style,label in [('coarse_F_lambda','#cf772f','--','2-dex abundance mesh'),('fine_F_lambda','#23659a','-','1-dex abundance mesh')]:
                    ax.plot(w[mask],100*(data[key][mask]/direct[mask]-1),color=color,ls=style,lw=.85,label=label)
        else:ax.text(.5,.45,'Missing qualified corner',ha='center',transform=ax.transAxes,color='.4',fontsize=11,bbox=dict(facecolor='white',edgecolor='none',pad=2))
        ax.set_xlim(3000,10000)
    for ax in axes[-1]:ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)',fontsize=15)
    fig.supylabel('Interpolation error (%)',fontsize=15,x=.02)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.10,right=.985,bottom=.085,top=.99,wspace=.19,hspace=.18)
    save(fig,args.output,'interpolation_residuals')
    (args.output/'plot_provenance.json').write_text(json.dumps(dict(selected_table_sha256=sha(args.inputs/'selected-requests.csv'),abundance_sequence=[dict(model_id=r['model_id'],log_H_He=h,spectrum_sha256=r['spectrum_sha256']) for h,r,_ in selected],flux_normalization='none',wavelength_interpolation='none'),indent=2)+'\n')
    print(json.dumps(dict(figures=3,sequence_spectra=11,requests=616,numerical=588)))

if __name__=='__main__':main()
