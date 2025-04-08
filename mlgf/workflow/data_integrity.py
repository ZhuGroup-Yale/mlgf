from mlgf.data import Data
from mlgf.lib.ml_helper import get_sigma_fit, gGW_mo_saiao

import numpy as np
import argparse
import os

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def check_integrity(mlf_chkfile, trace_thresh = 1e-8, trace_thresh_gw = 1e-5):
    """some data integrity checks to check the validity of saved electronic structure data

    Args:
        mlf_chkfile (str): chk file
        trace_thresh (float, optional): threshold for trace invariance. Defaults to 1e-8.
        trace_thresh_gw (float, optional): another threshold for trace invariance. Defaults to 1e-5.

    Returns:
        _type_: _description_
    """    
    file_return = None
    try:
        mlf = Data.load_chk(mlf_chkfile)
        dm_saiao = mlf['dm_saiao']
        if 'dm_gw' in mlf.keys():
            dm_gw = mlf['dm_gw']
        else:
            dm_gw = mlf['dm_cc']

        mo_occ = mlf['mo_occ']
        full_sigma = mlf['sigmaI']
        full_freqs = mlf['freqs']
        selected_freqs = mlf['omega_fit']

        C_ao_saiao = mlf['C_ao_saiao']
        mo_coeff = mlf['mo_coeff']
        S_ao = mlf['ovlp']

        C_saiao_mo = np.linalg.multi_dot((C_ao_saiao.T, S_ao, mo_coeff))
        C_mo_saiao = C_saiao_mo.T


        sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)
        sigma_saiao = gGW_mo_saiao(sigma_fit, C_mo_saiao)


        trace_dm_test = np.abs(np.trace(dm_saiao) - np.sum(mo_occ))
        trace_sigma_test = np.sum(np.abs(np.trace(sigma_saiao) - np.trace(sigma_fit)))

        if trace_dm_test > trace_thresh:
            print(f'Tr(dm_saiao) == particle_number test failed for mlf_chkfile: {mlf_chkfile}')
            file_return = mlf_chkfile
        if trace_sigma_test > trace_thresh:
            print(f'Tr(sigma_saiao(iw)) == Tr(sigma_mo(iw)) test failed for mlf_chkfile: {mlf_chkfile}')
            file_return = mlf_chkfile
        
        trace_dm_test = np.abs(np.trace(dm_gw) - np.sum(mo_occ))
        if trace_dm_test > trace_thresh_gw:
            print(trace_dm_test)
            print(f'Tr(dm_gw(iw)) == particle_number test failed for mlf_chkfile: {mlf_chkfile}')
            file_return = mlf_chkfile
            
    except Exception as e:
        print(f'ERROR CHECKING CHKFILE! : {mlf_chkfile}, {str(e)}')
        file_return = mlf_chkfile
    return file_return
    
def check_integrity_mpi(validation_dir, trace_thresh = 1e-8, trace_thresh_gw = 1e-2):
    validation_files = [f'{validation_dir}/{f}' for f in os.listdir(validation_dir) if '.chk' in f]
    indices = np.arange(len(validation_files))
    indices_subset = indices[indices % size == rank]
    validation_files_subset = [validation_files[i] for i in indices_subset]

    error_files = []

    for validation_file in validation_files_subset:
        
        error_file = check_integrity(validation_file, trace_thresh = trace_thresh, trace_thresh_gw = trace_thresh_gw)
        if not error_file is None:
            error_files.append(error_file)

    return error_files

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='data_integrity.py')
    # must have key jobs with a list of dictionaries, each with key model_file, output_file, mlf_chkfile
    parser.add_argument('--src', required=True, help='head directory where all molecules subdirectories have checkpoints from generate.py')

    args = parser.parse_args()
    src = args.src

    check_integrity_mpi(src)

    MPI.Finalize()



