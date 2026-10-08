#!/usr/bin/env python3
"""Fetch public CALSPEC hot-DA observations and separately labeled references."""
from pathlib import Path
import argparse,hashlib,json,urllib.request

PRODUCTS=('gd153_stiswfcnic_004.fits','g191b2b_stiswfcnic_004.fits',
          'gd153_mod_012.fits','g191b2b_mod_012.fits','gd153_stisnic_006.fits')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    record=[]
    for name in PRODUCTS:
        archive='calspec' if name == 'gd153_stisnic_006.fits' else 'current_calspec'
        url='https://ssb.stsci.edu/cdbs/'+archive+'/'+name
        destination=args.output/name
        if not destination.exists():
            with urllib.request.urlopen(url,timeout=60) as stream:
                payload=stream.read()
            if not payload.startswith(b'SIMPLE'):
                raise ValueError('Server did not return FITS data: '+url)
            destination.write_bytes(payload)
        record.append(dict(file=name,url=url,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            role=('model reference' if '_mod_' in name else
                  'older same-star STIS resolution metadata' if name == 'gd153_stisnic_006.fits' else
                  'composite observed SED; optical pedigree must be checked')))
        print(name,destination.stat().st_size,flush=True)
    (args.output/'sources.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    main()
