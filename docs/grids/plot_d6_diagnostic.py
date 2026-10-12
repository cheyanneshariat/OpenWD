"""Plot the verified native D6 spectrum from the diagnostic archive."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verification',type=Path,default=Path(__file__).with_name('diagnostics-2026-10-12')/'ARCHIVE_VERIFICATION.json');a=ap.parse_args();receipt=json.loads(a.verification.read_text());assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==receipt['archive_sha256']
 with tarfile.open(a.archive) as bank:
  for n,h in receipt['member_hashes'].items():assert hashlib.sha256(bank.extractfile(n).read()).hexdigest()==h
  rows=json.loads(bank.extractfile('native-models.json').read());accepted=[r for r in rows if r['status']=='COMPLETE_NUMERICAL'];assert len(rows)==4 and len(accepted)==1;r=accepted[0];assert r['family']=='D6' and r['audit']['certificate']['verified'] and r['audit']['independent_final_spectrum_source_verified'];raw=bank.extractfile(r['spectrum_archive_path']).read();assert hashlib.sha256(raw).hexdigest()==r['spectrum_sha256'];v=np.loadtxt(io.BytesIO(raw));assert np.isfinite(v).all() and (np.diff(v[:,0])>0).all() and (v[:,1]>=0).all()
 a.output.mkdir(parents=True,exist_ok=False);plt.style.use(Path(__file__).with_name('grid-style.mplstyle'));plt.rcParams.update({'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11});fig,ax=plt.subplots(figsize=(9,4));mask=(v[:,0]>=3000)&(v[:,0]<=10000);ax.plot(v[mask,0],v[mask,1]/1e7,color='#24496a',lw=.9);ax.set_xlim(float(v[mask,0].min()),float(v[mask,0].max()));ax.set_ylim(bottom=0);ax.set_title(r'D6, $T_{\rm eff}=15{,}680$ K, $\log g=6.3$');ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)');ax.set_ylabel(r'$F_\lambda$ ($10^7$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)');fig.subplots_adjust(left=.105,right=.985,bottom=.17,top=.89);assert not fig.texts and not fig.legends and not ax.texts and ax.get_legend() is None
 for ext in ['png','pdf']:fig.savefig(a.output/f'd6_spectrum.{ext}',dpi=250)
 plt.close(fig);(a.output/'plot_provenance.json').write_text(json.dumps(dict(archive_sha256=receipt['archive_sha256'],spectrum_sha256=r['spectrum_sha256'],flux_normalization='none',wavelength_resampling='none',physical_accuracy_verified=False,independent_depth_convergence_verified=False),indent=2)+'\n')
if __name__=='__main__':main()
