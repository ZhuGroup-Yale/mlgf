import os
import argparse
import joblib
import json
import gc

import torch
import numpy as np
import pandas as pd

from mlgf.model.pytorch.pt_alias import get_model_from_alias
from mlgf.model.pytorch.train import train_graph_model, train_refine_graph_model
from mlgf.model.active_learning import get_homo_lumo_uncertainties_train_examples

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def train_graph_ensemble_mpi(gnn_orch_file, seeds, train_config, initial_model_states = None, debug = False, safer_memory = False):
    """train an ensemble of MBGF-Net on mpi procs

    Args:
        gnn_orch_file (str): joblib file for GraphOrchestrator
        seeds (list[int]): random seeds for each proc
        train_config (dict): training configuration variables (epochs, batch_size, learning_rate, etc)
        initial_model_states (list[OrdererdDict], optional): list of previous model states to start the training of each from. Defaults to None.
        debug (bool, optional): more print statements if True. Defaults to False.
    """    
    out_tasks = []

    gnn_orch = joblib.load(gnn_orch_file)
    model_kwargs = gnn_orch.model_kwargs
    loss_kwargs = gnn_orch.loss_kwargs
    model_alias = getattr(gnn_orch, 'model_alias')
    
    for i in range(len(seeds)):
        if i % size == rank:
            out_tasks.append(i)
                    
    for i in out_tasks:
        pt_file = gnn_orch_file.replace('.joblib', f'_{i}.pt')        
        model = get_model_from_alias(model_alias, **model_kwargs)

        if initial_model_states is not None:
            model.load_state_dict(initial_model_states[i])
        if type(train_config) is list:
            for c, config in enumerate(train_config):
                if c != 0:
                    state_dict = torch.load(pt_file, map_location=torch.device('cpu'))
                    model.load_state_dict(state_dict)
                
                model_i = train_graph_model(gnn_orch.data, model, config, seed = seeds[i], loss_kwargs = loss_kwargs, rank = rank)
                torch.save(model_i.state_dict(), pt_file)
        else:
            log_dir = train_config.get('log_dir', None)
            model_i = train_graph_model(gnn_orch.data, model, train_config, seed = seeds[i], loss_kwargs = loss_kwargs, rank = rank, log_dir = log_dir)
            torch.save(model_i.state_dict(), pt_file)

        del model_i
        del model
        gc.collect()
        torch.cuda.empty_cache()

def refine_graph_ensemble_mpi(gnn_orch_file, seeds, train_config, uncertainty_sort = 'mean_max'):
    """Active refinement of MBGF-Net ensemble using uncertainty estimates

    Args:
        gnn_orch_file (str): joblib file for GraphOrchestrator
        seeds (list[int]): random seeds for each proc
        train_config (dict): training configuration variables (epochs, batch_size, learning_rate, etc)
        uncertainty_sort (str, optional): scalar aggregation metric to use for ranking unusal cases. Defaults to 'mean_max'.
    """    
    
    gnn_orch = joblib.load(gnn_orch_file)
    uncertainty_csv = gnn_orch_file.replace('.joblib', '_uncertainties.csv')

    if not os.path.isfile(uncertainty_csv):
        indices = np.arange(len(gnn_orch.data))
        indices_subset = indices[indices % size == rank]
        df = get_homo_lumo_uncertainties_train_examples(gnn_orch_file, indices_subset)
        comm.Barrier()
        gathered_dfs = comm.gather(df)
        
        if rank == 0:
            gathered_dfs = pd.concat(gathered_dfs)
            gathered_dfs = gathered_dfs.sort_values(by = uncertainty_sort, ascending = False)
            gathered_dfs.to_csv(uncertainty_csv, index = False)
        comm.Barrier() 
    df = pd.read_csv(uncertainty_csv)
    top_refine = train_config.get('top_refine', 200)
    indices_refine = list(df['index'])[:top_refine]

    out_tasks = []
    model_kwargs = gnn_orch.model_kwargs
    loss_kwargs = gnn_orch.loss_kwargs
    model_alias = getattr(gnn_orch, 'model_alias')
    for i in range(len(seeds)):
        if i % size == rank:
            out_tasks.append(i)
                    
    for i in out_tasks:
        pt_file = gnn_orch_file.replace('.joblib', f'_refined_{i}.pt')        
        print(train_config)
        model = get_model_from_alias(model_alias, **model_kwargs)

        model.load_state_dict(gnn_orch.model_states[i])

        log_dir = train_config.get('log_dir', None)

        model_i = train_refine_graph_model(gnn_orch.torch_data_root, indices_refine, model, train_config, seed = seeds[i], loss_kwargs = loss_kwargs, rank = rank, log_dir = log_dir)
        torch.save(model_i.state_dict(), pt_file)

        del model_i
        del model
        gc.collect()
        torch.cuda.empty_cache()

# regardless of the where the model was trained (cuda or cpu), load the state_dict on cpu
# this may be changed in the future, but we want to isolate GPU usage to model training for now
def collect_models(gnn_orch_file, n):
    """collect models into the pickled joblib GraphOrchestrator

    Args:
        gnn_orch_file (str): joblib file for GraphOrchestrator
        n (_type_): number of NN in the ensemble
    """    
    state_dicts = []
    gnn_orch = joblib.load(gnn_orch_file)
    for i in range(n):
        pt_file = gnn_orch_file.replace('.joblib', f'_{i}.pt')
        state_dict = torch.load(pt_file, map_location=torch.device('cpu'))
        state_dicts.append(state_dict)

    setattr(gnn_orch, 'model_states', state_dicts)
    
    gnn_orch.dump(gnn_orch_file)

def collect_models_refined(gnn_orch_file, n):
    """collect actively refined models into the pickled joblib GraphOrchestrator

    Args:
        gnn_orch_file (str): joblib file for GraphOrchestrator
        n (_type_): number of NN in the ensemble
    """    
    state_dicts = []
    gnn_orch = joblib.load(gnn_orch_file)
    for i in range(n):
        pt_file = gnn_orch_file.replace('.joblib', f'_refined_{i}.pt')
        state_dict = torch.load(pt_file, map_location=torch.device('cpu'))
        state_dicts.append(state_dict)

    setattr(gnn_orch, 'model_states', state_dicts)
    gnn_orch_file = gnn_orch_file.replace('.joblib', '_refined.joblib')
    gnn_orch.dump(gnn_orch_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_graph_ensemble.py')
    parser.add_argument('--json_spec', required=True, help='json file that has keys: gnn_orch_files (path string), train_files (list of lists same length as gnn_orch_files)')
    parser.add_argument('--transfer_model', action = 'store_true', help='Train a transfer model, a prepared gnn_orchestrator has to have a transfer_data_set attribute AND a model_states from the pretraied model to be transferred')
    parser.add_argument('--active_refinement', action = 'store_true', help='refine existing GNN ensemble with active learning')
    parser.add_argument('--debug', action = 'store_true', help='use the debug training function')
    parser.add_argument('--strong', action = 'store_true', help='magnitude splitting of self-energy')
    parser.add_argument('--torch_max_split', required = False, help='integer, env variable max_split_size_mb in MB')
    parser.add_argument('--safer_memory', required = False, action = 'store_true', help='safer memory use by GPU for large systems')

    args = parser.parse_args()
    json_spec = args.json_spec
    transfer_model = args.transfer_model
    active_refinement = args.active_refinement
    torch_max_split = args.torch_max_split

    if torch_max_split is not None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{torch_max_split}"

    assert('.json' in json_spec)
     
    with open(json_spec) as f:
        job = json.load(f)
    
    gnn_orch_file = job['gnn_orch_file']
    gnn_orch_file_to_copy = job.get('gnn_orch_file_to_copy', None)
    if rank == 0:

        outdir = '/'.join(gnn_orch_file.split('/')[:-1])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif not os.path.isdir(outdir):
                raise FileExistsError(f'{outdir} exists but is not a directory')
                comm.Abort()

        if gnn_orch_file_to_copy is not None:
            gnn_orch = joblib.load(gnn_orch_file_to_copy)
        else:
            gnn_orch = joblib.load(gnn_orch_file)

        gnn_orch.model_alias = job.get("model_alias", gnn_orch.model_alias)
        gnn_orch.model_kwargs = job.get("model_kwargs", gnn_orch.model_kwargs)
        gnn_orch.model_alias_transfer = job.get("model_alias_transfer", getattr(gnn_orch, 'model_alias_transfer', None))
        gnn_orch.model_kwargs_transfer = job.get("model_kwargs_transfer", getattr(gnn_orch, 'model_kwargs_transfer', {}))

        gnn_orch.loss_kwargs = job.get("loss_kwargs", gnn_orch.loss_kwargs)
        # gnn_orch.loss_kwargs['energy_scale'] = gnn_orch.scale_y
        gnn_orch.loss_kwargs['always_in_batch'] = job.get('always_in_batch', [])
        gnn_orch.ensemble_n = job.get("ensemble_n", gnn_orch.ensemble_n)
        joblib.dump(gnn_orch, gnn_orch_file)
    
    print(f'gnn_orch_file: {gnn_orch_file}', flush = True)
    train_config = job['train_config']
    ensemble_n = job.get('ensemble_n', 1)
    seeds = list(range(42, 42 + ensemble_n))
    initial_state_src = job.get('initial_state_src', None)
    if initial_state_src is not None:
        initial_model_states = joblib.load(initial_state_src).model_states
    else:
        initial_model_states = None

    comm.Barrier()
    train_graph_ensemble_mpi(gnn_orch_file, seeds, train_config, initial_model_states = initial_model_states, debug = args.debug, safer_memory = args.safer_memory)
    print(f'Rank {rank} finished training!', flush = True)
    comm.Barrier()

    comm.Barrier()
    if rank == 0:
        collect_models(gnn_orch_file, ensemble_n)
        
    comm.Barrier()
    train_config_active = job.get('train_config_active', None)
    if train_config_active is not None:
        refine_graph_ensemble_mpi(gnn_orch_file, seeds, train_config_active)
        comm.Barrier()
        if rank == 0:
            collect_models_refined(gnn_orch_file, ensemble_n)
