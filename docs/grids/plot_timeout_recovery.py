"""Plot saved retry coverage, DAB interpolation and the accepted DAO spectrum."""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig, output, name):
    for ext in ('png','pdf'): fig.savefig(output/f'{name}.{ext}',dpi=250)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--style',type=Path,default=Path(__file__).with_name('grid-style.mplstyle'))
    args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    plt.style.use(args.style)
    for line in (args.inputs/'INPUT_SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(args.inputs/name)==expected,name
    rows=list(csv.DictReader((args.inputs/'dab-selected-requests.csv').open()))
    assert len(rows)==len({r['model_id'] for r in rows})==616
    styles={'COMPLETE_NUMERICAL':('#237a57','o','Completed'),
            'DOMAIN_OR_UNSUPPORTED_CONFIG':('#a5a5a5','^','Unsupported'),
            'WORKER_CRASH_SIGSEGV':('#555555','x','Worker crash')}
    fig,axes=plt.subplots(3,4,figsize=(15,9.5),sharex=True,sharey=True)
    for ax,h in zip(axes.flat,range(-6,5)):
        subset=[r for r in rows if float(r['log_H_He'])==h];assert len(subset)==56
        for key,(color,marker,label) in styles.items():
            q=[r for r in subset if r['derived_status']==key]
            if q:ax.scatter([float(r['teff_K'])/1000 for r in q],[float(r['logg']) for r in q],color=color,marker=marker,s=26)
        n=sum(r['status']=='COMPLETE_NUMERICAL' for r in subset)
        ax.set_title(r'$\log_{10}[N(\mathrm{H})/N(\mathrm{He})]='+f'{h:g}'+r'$'+f' ({n}/56)',fontsize=11)
        ax.set_xlim(10.8,41.5);ax.set_ylim(9.7,5.8);ax.set_xticks([12,20,30,40]);ax.set_yticks([6,7,8,9])
    axes.flat[-1].set_visible(False)
    for ax in axes[-1,:3]:ax.set_xlabel(r'$T_{\rm eff}$ (kK)',fontsize=14)
    for ax in axes[:,0]:ax.set_ylabel(r'$\log g$ (cgs)',fontsize=14)
    handles=[Line2D([],[],ls='',marker=marker,color=color,label=label) for color,marker,label in styles.values()]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.055,right=.985,bottom=.09,top=.95,wspace=.12,hspace=.27)
    save(fig,args.output,'dab_coverage')

    report=json.loads((args.inputs/'interpolation/comparison.json').read_text())
    assert report['measured']==len(report['records'])==8
    fig,axes=plt.subplots(4,2,figsize=(11,11.7),sharex=True)
    for ax,row in zip(axes.flat,report['records']):
        p=row['parameters'];ax.text(.97,.94,rf'$T_{{\rm eff}}={p["teff_K"]/1000:g}\,\mathrm{{kK}},\ \log g={p["logg"]:g}$'+'\n'+rf'$\log_{{10}}[N(\mathrm{{H}})/N(\mathrm{{He}})]={p["log_H_He"]:g}$',ha='right',va='top',transform=ax.transAxes,fontsize=11,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
        ax.axhline(0,color='.65',lw=.7)
        for y in (-3,3):ax.axhline(y,color='.65',ls=':',lw=.6)
        with np.load(args.inputs/'interpolation'/f'{row["model_id"]}.npz',allow_pickle=False) as data:
            w=data['wavelength_angstrom'];direct=data['independent_F_lambda'];mask=(w>=3000)&(w<=10000)
            assert np.isfinite(direct).all() and (direct>0).all()
            for key,color,style,label in [('coarse_F_lambda','#cf772f','--','2-dex abundance mesh'),('fine_F_lambda','#23659a','-','1-dex abundance mesh')]:
                ax.plot(w[mask],100*(data[key][mask]/direct[mask]-1),color=color,ls=style,lw=.85,label=label)
        ax.set_xlim(3000,10000)
    for ax in axes[-1]:ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)',fontsize=15)
    fig.supylabel('Interpolation error (%)',fontsize=15,x=.02)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.10,right=.96,bottom=.085,top=.99,wspace=.19,hspace=.18)
    save(fig,args.output,'interpolation_residuals')

    requests=list(csv.DictReader((args.inputs/'requests.csv').open()))
    dao=[r for r in requests if r['family']=='DAO' and r['status']=='COMPLETE_NUMERICAL'];assert len(dao)==1
    row=dao[0];spectrum=args.bank/row['spectrum_path'];assert sha(spectrum)==row['spectrum_sha256']
    data=np.loadtxt(spectrum);assert data.shape[1]==2 and np.isfinite(data).all() and (data[:,1]>=0).all() and (np.diff(data[:,0])>0).all()
    w,f=data.T;optical=(w>=3800)&(w<=7000)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    axes[0].plot(w,w*f,color='#23659a',lw=1.1,label='openWD')
    axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xlim(w[0],w[-1])
    axes[0].set_ylabel(r'$\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)')
    axes[1].plot(w[optical],f[optical]/1e8,color='#23659a',lw=1.1)
    axes[1].set_xlim(3800,7000);axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
    for ax in axes:ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)')
    axes[0].legend(frameon=False,fontsize=12)
    cfg=json.loads(row['config_json'])
    fig.suptitle(r'DAO: $T_{\rm eff}='+f'{cfg["effective_temperature"]/1000:g}'+r'\,\mathrm{kK},\ \log g='+f'{cfg["logg"]:g}'+r',\ \log_{10}[N(\mathrm{H})/N(\mathrm{He})]='+f'{cfg["log_hydrogen_to_helium"]:g}'+r'$',fontsize=14,y=.98)
    fig.subplots_adjust(left=.09,right=.96,bottom=.18,top=.84,wspace=.34)
    save(fig,args.output,'dao_spectrum')
    (args.output/'plot_provenance.json').write_text(json.dumps(dict(dab_table_sha256=sha(args.inputs/'dab-selected-requests.csv'),requests_sha256=sha(args.inputs/'requests.csv'),comparison_sha256=sha(args.inputs/'interpolation/comparison.json'),dao_spectrum_sha256=sha(spectrum),flux_normalization='none',wavelength_interpolation='none'),indent=2)+'\n')

if __name__=='__main__':main()
