import argparse
import os
from pathlib import Path
import warnings
import joblib
import pyscf
from mlgf.data import Data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='datacompat.py')
    parser.add_argument('SRCDIR', help='source directory')
    parser.add_argument('--upgrade-joblibs', action='store_true', help='upgrade joblib files, rather than converting to internal format')
    args = parser.parse_args()
    
    srcpath = Path(args.SRCDIR)
    if not srcpath.is_dir():
        raise ValueError(f'{srcpath} is not a directory')
    
    print(srcpath)
    joblibs = list(srcpath.glob('*.joblib'))
    
    # e.g. ['mol0', 'mol1' ...]
    stems = [Path(f).stem for f in joblibs]
    
    pyscf_checkfiles = [srcpath.joinpath(f'{stem}.chk') for stem in stems]
    custom_checkfiles = [srcpath.joinpath(f'{stem}.h5') for stem in stems]
    
    if not args.upgrade_joblibs:
        for j, chk, cchk in zip(joblibs, pyscf_checkfiles, custom_checkfiles):
            mdt_file_path = srcpath.joinpath(f'{j.stem}.mdt')
            dic = joblib.load(j)
            mdt = Data(dic)
            
            mol = pyscf.lib.chkfile.load_mol(chk)
            pyscfdata = pyscf.lib.chkfile.load(chk, 'scf')
            
            mdt.mol = mol
            mdt.e_tot = pyscfdata['e_tot']
            mdt.mo_energy = pyscfdata['mo_energy']
            mdt.mo_occ = pyscfdata['mo_occ']
            mdt.mo_coeff = pyscfdata['mo_coeff']
            mdt.save(mdt_file_path)