import os
import numpy as np
import argparse
import joblib
import time
import warnings
import psutil
import json

from mlgf.model.sklearn_model import SKL_Model
from mlgf.data import Dataset, Data

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def train_skl_model(model_file, train_files, feat_list_ii, feat_list_ij, target, exclude_core, coulomb_screen, coulomb_screen_basis,
ij_chunk_size, max_mem = None, dset_file = None):

    if max_mem is None: max_mem = int(psutil.virtual_memory().available / 1e6 / 2)
    if dset_file is None:
        dyn_imag_freq_points = 1j*np.array([1e-3, 0.1, 0.2, 0.5, 1.0, 2.0])
        dset = Dataset.from_files(train_files, data_format='chk')
        dset = dset.prep_dyn_features(dyn_imag_freq_points)
    else:
        dset = joblib.load(dset_file)

    # explicit garbage collection
    import gc
    gc.collect()

    t0 = time.time()
    model = SKL_Model(feat_list_ii, feat_list_ij, target, exclude_core = exclude_core, coulomb_screen_tol = coulomb_screen, coulomb_screen_basis = coulomb_screen_basis, 
    max_mem = max_mem, ij_chunk_size = ij_chunk_size, dyn_imag_freq_points = dyn_imag_freq_points)
    model.fit(dset)
    t = time.time() - t0
    print(f'Fitting took {t:0.2f}s')
    model.dump(model_file)

def train_model(job):
    model_file = job['model_file']
    train_files = job['train_files']
    feat_list_ii = job.get('feat_list_ii', defaults['feat_list_ii'])
    feat_list_ij = job.get('feat_list_ij', defaults['feat_list_ij'])
    target = job.get('target', defaults['target'])
    coulomb_screen_basis = job.get('coulomb_screen_basis', defaults['coulomb_screen_basis'])
    exclude_core = job.get('exclude_core', defaults['exclude_core'])
    coulomb_screen = job.get('coulomb_screen', defaults['coulomb_screen'])
    max_mem = job.get('max_mem', defaults['max_mem'])
    dset_file = job.get('dset_file', None)
    ij_chunk_size = job.get('ij_chunk_size', defaults['ij_chunk_size'])
    train_skl_model(model_file, train_files, feat_list_ii, feat_list_ij, target, exclude_core, 
    coulomb_screen, coulomb_screen_basis, ij_chunk_size, max_mem = max_mem, dset_file = dset_file)

def train_models_mpi(jobs, defaults):

    out_tasks = []
    
    for i in range(len(jobs)):
        if i % size == rank:
            out_tasks.append(i)
    
    for i in out_tasks:
        train_model(jobs[i])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_krr.py')
    parser.add_argument('--json_spec', required=True, help='json file that has keys: model_files (path string), train_files (list of lists same length as model_files)')
    parser.add_argument('--no_new_folders', action = 'store_true', help='Do not go through all the jobs and make output folders from eacb output_file if they do not exist')

    args = parser.parse_args()
    json_spec = args.json_spec
    no_new_folders = args.no_new_folders
    defaults = {'exclude_core' : False, 'coulomb_screen_basis' : 'saiao', 'coulomb_screen' : 0.05, 'target': 'sigma_saiao',
         'feat_list_ii' : ["dm_saiao", "fock_saiao", "hcore_saiao", "vj_saiao", "vxc_saiao", "gf_dyn", "hyb_dyn"],
     'feat_list_ij' : ["dm_saiao", "fock_saiao", "hcore_saiao", "vj_saiao", "vxc_saiao", "gf_dyn", "hyb_dyn_off"],
    'ij_chunk_size' : 1e3, 'max_mem' : 5000}

    assert('.json' in json_spec)
     
    with open(json_spec) as f:
        spec = json.load(f)

    if 'jobs' in spec.keys():
        jobs =  spec['jobs']
    else:
        jobs = [spec]

    if rank == 0 and not no_new_folders:
        for job in jobs:
            outdir = '/'.join(job['model_file'].split('/')[:-1])
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            elif not os.path.isdir(outdir):
                raise FileExistsError(f'{outdir} exists but is not a directory')
                comm.Abort()
        
    
    train_models_mpi(jobs, defaults)
