import os
import numpy as np
import argparse
import joblib
import pandas as pd

from mlgf.workflow.get_ml_info import predict_sigma
from mlgf.lib.ml_helper import sigma_lo_mo, rotate_sigma_uncertainty
from mlgf.model.pytorch.data import reconstruct_rank2
from mlgf.data import Dataset, Data

def get_homo_lumo_uncertainties_train_examples(model_file, indices, device = None):
    """get MO basis uncertainty for training examples

    Args:
        model_file (_type_): GraphOrchestrator joblib file
        indices (list): indices of train data to use
        device (str, optional): gpu or cpu for acceleration. Defaults to None.

    Returns:
        pd.DataFrame: table with the uncertainty metrics
    """    
    max_errors = []
    max_mean_errors = []
    mean_max_errors = []
    fnames = []
    gnn_orch = joblib.load(model_file)
    for index in indices:
        sigma_uncertainty = gnn_orch.uncertainty_training_example(index, device = device)
        lumo_ind = gnn_orch.data[index].lumo_ind
        homo_ind = lumo_ind - 1
        C_lo_mo = reconstruct_rank2(gnn_orch.data[index].C_lo_mo, gnn_orch.data[index].nmo) 
        sigma_uncertainty = rotate_sigma_uncertainty(sigma_uncertainty, C_lo_mo)
        relative_error = np.zeros((4, gnn_orch.data[index].nomega))

        # not relative errors 
        relative_error[0, :] = sigma_uncertainty[homo_ind, homo_ind, :].real#/sigma[homo_ind, homo_ind, :].real
        relative_error[1, :] = sigma_uncertainty[homo_ind, homo_ind, :].imag#/sigma[homo_ind, homo_ind, :].imag
        relative_error[2, :] = sigma_uncertainty[lumo_ind, lumo_ind, :].real#/sigma[lumo_ind, lumo_ind, :].real
        relative_error[3, :] = sigma_uncertainty[lumo_ind, lumo_ind, :].imag#/sigma[lumo_ind, lumo_ind, :].imag
        relative_error = np.abs(relative_error)

        max_errors.append(np.max(relative_error))
        max_mean_errors.append(np.max(np.mean(relative_error, axis = 0)))
        mean_max_errors.append(np.mean(np.max(relative_error, axis = 0)))
        fnames.append(gnn_orch.data[index].fname)

    df = pd.DataFrame({'index' : indices, 'file_name' : fnames, 'max' : max_errors, 'max_mean' : max_mean_errors, 'mean_max' : mean_max_errors})
    return df

def get_homo_lumo_uncertainties(model_file, chkfiles):
    """get MO basis uncertainty for DFT calculations in chkfiles

    Args:
        model_file (_type_): GraphOrchestrator joblib file
        chkfiles (list): list of files with DFT calc stored
    Returns:
        pd.DataFrame: table with the uncertainty metrics
    """    
    max_errors = []
    max_mean_errors = []
    mean_max_errors = []
    for chkfile in chkfiles:
        sigma, sigma_uncertainty = predict_sigma(model_file, chkfile, return_uncertainty = True)
        mlf = Data.load_chk(chkfile)
        lumo_ind = mlf['nocc']
        homo_ind = lumo_ind - 1
        C_lo_mo = mlf['C_saiao_mo']
        sigma_uncertainty = rotate_sigma_uncertainty(sigma_uncertainty, C_lo_mo)
        sigma = sigma_lo_mo(sigma, C_lo_mo)
        relative_error = np.zeros((4, sigma.shape[-1]))

        # not relative errors 
        relative_error[0, :] = sigma_uncertainty[homo_ind, homo_ind, :].real#/sigma[homo_ind, homo_ind, :].real
        relative_error[1, :] = sigma_uncertainty[homo_ind, homo_ind, :].imag#/sigma[homo_ind, homo_ind, :].imag
        relative_error[2, :] = sigma_uncertainty[lumo_ind, lumo_ind, :].real#/sigma[lumo_ind, lumo_ind, :].real
        relative_error[3, :] = sigma_uncertainty[lumo_ind, lumo_ind, :].imag#/sigma[lumo_ind, lumo_ind, :].imag
        relative_error = np.abs(relative_error)

        max_errors.append(np.max(relative_error))
        max_mean_errors.append(np.max(np.mean(relative_error, axis = 0)))
        mean_max_errors.append(np.mean(np.max(relative_error, axis = 0)))

    df = pd.DataFrame({'file_name' : chkfiles, 'max' : max_errors, 'max_mean' : max_mean_errors, 'mean_max' : mean_max_errors})
    return df

if __name__ == '__main__':
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
    
    parser = argparse.ArgumentParser(prog='make_skl_model_mpi.py')
    parser.add_argument('--new_data_dir', required=True, help='place to get uncertainty of new DFT calculations')
    parser.add_argument('--gnn_orch_file', required=True, help="gnn orchestrator file that has the model to use for predicting self-energy")
    parser.add_argument('--output_csv', type=str, default="active_learning.csv", help="csv to write outputs to")
    
    args = parser.parse_args()
    gnn_orch_file = args.gnn_orch_file
    new_data_dir = args.new_data_dir
    output_csv = args.output_csv
    
    # queue all the files for uncertainty prediction
    all_mlf_chkfiles = [f'{new_data_dir}/{f}' for f in os.listdir(new_data_dir) if '.chk' in f]    
    
    indices = np.arange(len(all_mlf_chkfiles))
    indices_subset = indices[indices % size == rank]
    chkfiles_subset = [all_mlf_chkfiles[i] for i in indices_subset]
    print(f'rank {rank} has {len(chkfiles_subset)} jobs')

    comm.Barrier()
    df = get_homo_lumo_uncertainties(gnn_orch_file, chkfiles_subset)
    print(f'Rank {rank} finished collection into dataframe!')
    comm.Barrier()
    gathered_dfs = comm.gather(df)
    if rank == 0:
        gathered_dfs = pd.concat(gathered_dfs)
        gathered_dfs.to_csv(output_csv, index = False)
    comm.Barrier() 






        