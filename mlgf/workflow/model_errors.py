from mlgf.lib.dos_helper import get_dos_hf
from mlgf.lib.dm_helper import get_dipole, scalar_quadrupole, dm_mo_to_ao
import os
import numpy as np
import pandas as pd
import joblib
import argparse
import json
import h5py
from pyscf import lib
from pyscf import dft

def get_hlb(validation_output, energy_levels):
    homo = validation_output[energy_levels][validation_output['nocc']-1]
    lumo = validation_output[energy_levels][validation_output['nocc']]
    bg = lumo - homo
    return np.array([homo, lumo, bg])

def get_dos_mre(dos_ml, dos_true):
    return np.sum(np.abs(dos_ml-dos_true))/np.sum(dos_true)

# model_predictions top directory, json spec of folders to iterate
def validations_to_table(validation_files, validation_names, do_dm = False):
    validation_data = []
    data_dict = {'file_name' : [], 'validation_name' : [], 'MF HOMO' : [], 'MF LUMO' : [], 'MF BG' : [], 'ML HOMO' : [], 'ML LUMO' : [],
    'ML BG' : [], 'True HOMO' : [], 'True LUMO' : [], 'True BG' : [], 'ML DOS Error': [], 'MF DOS Error' : []}
    if do_dm:
        data_dict['True Dipole'] = []
        data_dict['ML Dipole'] = []
        data_dict['MF Dipole'] = []
        data_dict['True Quadrupole'] = []
        data_dict['ML Quadrupole'] = []
        data_dict['MF Quadrupole'] = []

    for validation_file in validation_files:
        compute_reference_vals = True
        for validation_name in validation_names:
            out = lib.chkfile.load(validation_file, validation_name)
            if out is None:
                continue

            if compute_reference_vals:
                mf = get_hlb(out, 'mo_energy')
                true = get_hlb(out, 'qpe_true')
                dos_hf = get_dos_hf(out['mo_energy'], out['freqs'], out['eta'])
                dos_error_mf = get_dos_mre(dos_hf, out['dos_true'])

            ml = get_hlb(out, 'qpe_ml')
            dos_error_ml = get_dos_mre(out['dos_ml'], out['dos_true'])

            data_dict['MF HOMO'].append(mf[0])
            data_dict['MF LUMO'].append(mf[1])
            data_dict['MF BG'].append(mf[2])

            data_dict['ML HOMO'].append(ml[0])
            data_dict['ML LUMO'].append(ml[1])
            data_dict['ML BG'].append(ml[2])

            data_dict['True HOMO'].append(true[0])
            data_dict['True LUMO'].append(true[1])
            data_dict['True BG'].append(true[2])

            data_dict['MF DOS Error'].append(dos_error_mf)
            data_dict['ML DOS Error'].append(dos_error_ml)

            data_dict['file_name'].append(validation_file)
            data_dict['validation_name'].append(validation_name)

            if do_dm:
                # print(f'computing dipoles on rank {rank} for {validation_file}')
                if compute_reference_vals:
                    mlf_chkfile = out['mlf_chkfile']
                    if type(mlf_chkfile) is bytes:
                        mlf_chkfile = mlf_chkfile.decode('utf-8')
                    mlf = lib.chkfile.load(mlf_chkfile, 'mlf')
                    mol = lib.chkfile.load_mol(mlf_chkfile)

                    rks = dft.RKS(mol)
                    rks.xc = mlf['xc']
                    if type(rks.xc) is bytes:
                        rks.xc = rks.xc.decode('utf-8')
                    scf_data = lib.chkfile.load(mlf_chkfile, 'scf')
                    rks.__dict__.update(scf_data)
                    
                    mo_coeff = mlf['mo_coeff']
                    dm_true = dm_mo_to_ao(out['dm_true'], mo_coeff)
                    dm_mf = mlf['dm_hf']
                    dipole_true = get_dipole(rks, dm_true)
                    dipole_mf = get_dipole(rks, dm_mf)
                
                dm_ml = dm_mo_to_ao(out['dm_ml'], mo_coeff)
                dipole_ml = get_dipole(rks, dm_ml)
                data_dict['True Dipole'].append(dipole_true)
                data_dict['ML Dipole'].append(dipole_ml)
                data_dict['MF Dipole'].append(dipole_mf)

                data_dict['True Quadrupole'].append(scalar_quadrupole(mol, dm_true))
                data_dict['ML Quadrupole'].append(scalar_quadrupole(mol, dm_ml))
                data_dict['MF Quadrupole'].append(scalar_quadrupole(mol, dm_mf))

            compute_reference_vals = False

   
    df_perf_mat = pd.DataFrame(data_dict)
    return df_perf_mat

if __name__ == '__main__':
    try: 
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        rank, size = 0, 1
        comm = None
    # configuration variables at the top
    default_basis = 'saiao'
    parser = argparse.ArgumentParser(prog='model_errors.py')
    parser.add_argument('--output_csv', required = True, help='csv to write file')
    parser.add_argument('--json_spec', required = True, help='must have keys models (list of models with named directories in mpsrc top directory), and train_files (model.train_files)')

    args = parser.parse_args()    

    with open(args.json_spec) as f:
        spec = json.load(f)

    output_dir = spec['validation_dir']

    if not args.output_csv is None:
        do_dm = 'm' in spec['properties']
        if type(output_dir) == list:
            df_list = []
            for i in range(len(output_dir)):
                df_new = validations_to_table(output_dir[i], spec['model_files'], do_dm = do_dm)
                df_list.append(df_new.copy())
            df = pd.concat(df_list)
            
        else:    
            df = validations_to_table(output_dir, spec['model_files'], do_dm = do_dm)
        print(f'Rank {rank} finished collection into dataframe!')
        comm.Barrier()
    
        gathered_dfs = comm.gather(df)
        if rank == 0:
            gathered_dfs = pd.concat(gathered_dfs)
            gathered_dfs.to_csv(args.output_csv, index = False)
        comm.Barrier() 

    MPI.Finalize()