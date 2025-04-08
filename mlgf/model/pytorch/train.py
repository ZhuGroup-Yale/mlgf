from mlgf.model.pytorch.pt_alias import get_model_from_alias
from mlgf.model.pytorch.loss import SigmaLoss
from mlgf.model.pytorch.data import construct_symmetric_tensor, reconstruct_rank2
from mlgf.model.pytorch.helpers import setup_seed
from mlgf.model.gnn_orchestrator import GraphDataset
from mlgf.model.pytorch.helpers import count_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from mlgf.data import Data
from mlgf.workflow.get_ml_info import get_properties_cc
from torch_geometric.loader import DataLoader
import torch_geometric

# from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CyclicLR, ExponentialLR, LambdaLR, ChainedScheduler, CosineAnnealingWarmRestarts

from torch.func import stack_module_state
from torch.func import functional_call
from torch import vmap

import os
import numpy as np
import argparse
import joblib
import time
import warnings
import psutil
import json
import copy
import random
import gc 
import multiprocessing as mp
import threadpoolctl
from functools import partial

def get_lr_decay_interval(epochs, n_batches, decay_rate, lr_i, lr_f):
    """get the learning rate decay interval given an initial and final rate

    Args:
        epochs (int)
        n_batches (int)
        decay_rate (float): decay by what proportion each batch
        lr_i (float): initial rate
        lr_f (float): final rate

    Returns:
        int : decay step interval
    """    
    return int(epochs*n_batches*np.log(decay_rate)/np.log(lr_f/lr_i))

def train_graph_model(data, model, train_config, seed = 42, loss_kwargs = {}, rank = 0, log_dir = None): 
    """primary training function for GNN

    Args:
        data (GraphDataset object): training data object
        model (GNN): initialized GNN model
        train_config (dict): configuration variables for training
        seed (int, optional): random seed for setup_seed(). Defaults to 42.
        loss_kwargs (dict, optional): loss keyword argmuents beta_i and FMO definition. Defaults to {}.
        rank (int, optional): MPI rank for printing/debugging and training multiple ensembles. Defaults to 0.
        log_dir (str, optional): unused, if you want to use tensorboard logging, the SummaryWriter can dump here. Defaults to None.

    Returns:
        trained GNN
    """    
    setup_seed(seed)
    torch_geometric.seed.seed_everything(seed)
    # assert len(data_x) == len(data_y)
    device =  train_config.get('device', None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('device cuda requested but cuda not available, using CPU')
        device = 'cpu'

    if rank == 0:
        print('train_config: ', train_config, flush = True)
        print('loss_kwargs: ', loss_kwargs, flush = True)
        print('Device used for training: ', device, flush = True)
        total_num_params = count_parameters(model)
        print('Model total number of params: ', total_num_params, flush = True)

    model.to(device)
    freqs_dos = np.linspace(-1, 1, 201)
    
    n_epochs = int(train_config.get('epochs', None))
    if n_epochs is None:
        n_epochs = int(train_config.get('steps', 1000))
        
    weight_decay = train_config.get('weight_decay', 0.0)
    learning_rate = train_config.get('learning_rate', 0.001)
    batch_size = train_config.get('batch_size', np.inf) # the number of molecules per backward pass
    ndata = len(data)
    batch_size = min(batch_size, ndata) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    shuffle = train_config.get('shuffle', False)
    num_workers = train_config.get('num_workers', 0)
    pin_memory = train_config.get('pin_memory', True)

    # the learning rate scheduler params, by default turned off
    cyclic_range = train_config.get('cyclic_range', [-1, -2]) 
    exp_range = train_config.get('exp_range', [-1, -2]) 
    cos_range = train_config.get('cos_range', [-1, -2]) 
    n_cycles = train_config.get('lr_cycles', 1.0)
    steps_per_cycle = (cyclic_range[1] - cyclic_range[0])/(2*n_cycles)
    learning_rate_deacy_rate = train_config.get('learning_rate_decay_rate', 1.0)
    cycle_lr_min = train_config.get('cycle_lr_min', learning_rate)
    cycle_lr_max = train_config.get('cycle_lr_max', learning_rate)
    cosine_t0 = train_config.get('cosine_t0', n_epochs) # number of epochs per restart
    granular_print = train_config.get('granular_print', False) # number of epochs per restart
    conformer_batch = train_config.get('conformer_batch', 1) # number of epochs per restart

    if conformer_batch == 1:
        conformer_alpha = train_config.get('conformer_alpha', 0) # number of epochs per restart
    else:
        conformer_alpha = 1. / (conformer_batch + 1)


    do_conformers = conformer_alpha > 0

    cyclic_scheduler = CyclicLR(optimizer, base_lr=cycle_lr_min, max_lr=cycle_lr_max, step_size_up = steps_per_cycle, step_size_down = steps_per_cycle, cycle_momentum=False)
    exponential_scheduler = ExponentialLR(optimizer, gamma=learning_rate_deacy_rate)
    cos_scheduler = CosineAnnealingWarmRestarts(optimizer, cosine_t0)

    loss_kwargs['rank'] = rank
    loss_kwargs['device'] = device
    loss_kwargs['batch_size'] = batch_size
    loss_kwargs['omega_fit'] = data[0].iomega
    
    criterion = SigmaLoss(**loss_kwargs)
    criterion = criterion.to(device)

    # if not log_dir is None and rank == 0:
    #     writer = SummaryWriter(log_dir=log_dir)

    loader = DataLoader(data, batch_size = batch_size, shuffle = shuffle, pin_memory=pin_memory, num_workers = num_workers) #, worker_init_fn = seed
    training_scale = getattr(model, 'scale', 1.0)
    print(f'training_scale: {training_scale}', flush = True)
    for epoch in range(n_epochs):
        t0 = time.time()
        for b, batch in enumerate(loader):
            optimizer.zero_grad()   
            loss = 0
            batch = batch.to(device) 
            
            for i in range(len(batch)):
                mol = batch[i]
                sigma_ml = model(mol, train = True)
                sigma_true = construct_symmetric_tensor(mol.sigma_ii, mol.sigma_ij, mol)/training_scale
                loss += criterion(sigma_ml, sigma_true, C_lo_mo = mol.get('C_lo_mo', None), homo_ind = mol.get('homo_ind', None), lumo_ind = mol.get('lumo_ind', None), nmo = mol.nmo, mo_energy = mol.get('mo_energy', None), energy_scale = training_scale)

            loss.backward()
            optimizer.step()    
        
        # write log
        if rank == 0:
            t1 = time.time()
            t = t1-t0
            if loss != 0:
                loss_val = loss.detach().cpu().numpy()
            else:
                loss_val = -1 # means the loss was not computed in the final batch
            step_size = optimizer.param_groups[0]['lr']
            print(f'time for Adam #{epoch}: {t:0.2f}s, main loop value (final batch loss): {loss_val:0.6e}, step_size: {step_size:0.6e}', flush = True)
            
            # write a log file for tensorboard
            # if not log_dir is None: 
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             writer.add_histogram(name + '/values', param.data, global_step=epoch)
            #             writer.add_histogram(name + '/grad', param.grad, global_step=epoch)  

        # learning rate scheduling
        if cyclic_range[0] <= epoch < cyclic_range[1]:
            cyclic_scheduler.step()
        if exp_range[0] <= epoch < exp_range[1]:
            exponential_scheduler.step()
        if cos_range[0] <= epoch < cos_range[1]:
            cos_scheduler.step()

    return model

def train_graph_model_lbfgs(data, model, train_config, seed = 42, loss_kwargs = {}, rank = 0, log_dir = None): 
    """LBGFS optimizer instead of Adam (not recommended)

    Args:
        same as train_graph_model()

    Returns:
        trained GNN
    """    
    setup_seed(seed)
    torch_geometric.seed.seed_everything(seed)
    # assert len(data_x) == len(data_y)
    device =  train_config.get('device', None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('device cuda requested but cuda not available, using CPU')
        device = 'cpu'

    if rank == 0:
        print('train_config: ', train_config, flush = True)
        print('loss_kwargs: ', loss_kwargs, flush = True)
        print('Device used for training: ', device, flush = True)
        total_num_params = sum(param.numel() for param in model.parameters())
        print('Model total number of params: ', total_num_params, flush = True)

    model.to(device)
    freqs_dos = np.linspace(-1, 1, 201)
    
    n_epochs = int(train_config.get('epochs', None))
    if n_epochs is None:
        n_epochs = int(train_config.get('steps', 1000))
        
    weight_decay = train_config.get('weight_decay', 0.0)
    learning_rate = train_config.get('learning_rate', 0.001)
    batch_size = train_config.get('batch_size', np.inf) # the number of molecules per backward pass
    history_size = train_config.get('history_size', 10)
    ndata = len(data)
    batch_size = min(batch_size, ndata) 

    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, history_size=history_size, max_iter=n_epochs)
    shuffle = train_config.get('shuffle', False)
    num_workers = train_config.get('num_workers', 0)
    pin_memory = train_config.get('pin_memory', True)

    # the learning rate scheduler params, by default turned off
    cyclic_range = train_config.get('cyclic_range', [-1, -2]) 
    exp_range = train_config.get('exp_range', [-1, -2]) 
    cos_range = train_config.get('cos_range', [-1, -2]) 
    n_cycles = train_config.get('lr_cycles', 1.0)
    steps_per_cycle = (cyclic_range[1] - cyclic_range[0])/(2*n_cycles)
    learning_rate_deacy_rate = train_config.get('learning_rate_decay_rate', 1.0)
    cycle_lr_min = train_config.get('cycle_lr_min', learning_rate)
    cycle_lr_max = train_config.get('cycle_lr_max', learning_rate)
    cosine_t0 = train_config.get('cosine_t0', n_epochs) # number of epochs per restart
    granular_print = train_config.get('granular_print', False) # number of epochs per restart
    conformer_batch = train_config.get('conformer_batch', 1) # number of epochs per restart

    if conformer_batch == 1:
        conformer_alpha = train_config.get('conformer_alpha', 0) # number of epochs per restart
    else:
        conformer_alpha = 1. / (conformer_batch + 1)


    do_conformers = conformer_alpha > 0

    cyclic_scheduler = CyclicLR(optimizer, base_lr=cycle_lr_min, max_lr=cycle_lr_max, step_size_up = steps_per_cycle, step_size_down = steps_per_cycle, cycle_momentum=False)
    exponential_scheduler = ExponentialLR(optimizer, gamma=learning_rate_deacy_rate)
    cos_scheduler = CosineAnnealingWarmRestarts(optimizer, cosine_t0)

    loss_kwargs['rank'] = rank
    loss_kwargs['device'] = device
    loss_kwargs['batch_size'] = batch_size
    loss_kwargs['omega_fit'] = data[0].iomega
    
    criterion = SigmaLoss(**loss_kwargs)
    criterion = criterion.to(device)

    loader = DataLoader(data, batch_size = batch_size, shuffle = shuffle, pin_memory=pin_memory, num_workers = num_workers) #, worker_init_fn = seed
    training_scale = getattr(model, 'scale', 1.0)
    print(f'training_scale: {training_scale}', flush = True)
    for epoch in range(n_epochs):
        t0 = time.time()
        for b, batch in enumerate(loader):

            batch = batch.to(device) 
            
            def closure():
                optimizer.zero_grad()
                loss = 0
                for i in range(len(batch)):
                    mol = batch[i]
                    sigma_ml = model(mol, train = True)
                    sigma_true = construct_symmetric_tensor(mol.sigma_ii, mol.sigma_ij, mol)/training_scale
                    loss += criterion(sigma_ml, sigma_true, C_lo_mo = mol.get('C_lo_mo', None), homo_ind = mol.get('homo_ind', None), lumo_ind = mol.get('lumo_ind', None), nmo = mol.nmo, mo_energy = mol.get('mo_energy', None), energy_scale = training_scale)
                loss.backward()
                return loss
            
            optimizer.step(closure)    
        
        # write log
        if rank == 0:
            t1 = time.time()
            t = t1-t0
            step_size = optimizer.param_groups[0]['lr']
            print(f'time for LBFGS #{epoch}: {t:0.2f}s, step_size: {step_size:0.6e}', flush = True)

        # learning rate scheduling
        if cyclic_range[0] <= epoch < cyclic_range[1]:
            cyclic_scheduler.step()
        if exp_range[0] <= epoch < exp_range[1]:
            exponential_scheduler.step()
        if cos_range[0] <= epoch < cos_range[1]:
            cos_scheduler.step()
            
    return model

def train_refine_graph_model(data_root, indices_refine, model, train_config, seed = 42, loss_kwargs = {}, rank = 0, log_dir = None): 
    """active refinement protocol in paper

    Args:
        data_root (torch data source): the directory where the GraphDataset is stored
        indices_refine (list[int]): indicies of molecules of highest uncertainty score

    Returns:
       trained GNN
    """    
    setup_seed(seed)
    torch_geometric.seed.seed_everything(seed)
    device =  train_config.get('device', None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('device cuda requested but cuda not available, using CPU')
        device = 'cpu'

    if rank == 0:
        print('loss_kwargs: ', loss_kwargs)
        print('Device used for training: ', device)
        total_num_params = sum(param.numel() for param in model.parameters())
        print('Model total number of params: ', total_num_params)

    model.to(device)
    freqs_dos = np.linspace(-1, 1, 201)
    
    n_epochs = int(train_config.get('epochs', None))
    if n_epochs is None:
        n_epochs = int(train_config.get('steps', 1000))
        
    weight_decay = train_config.get('weight_decay', 0.0)
    learning_rate = train_config.get('learning_rate', 0.001)
    batch_size = train_config.get('batch_size', np.inf) # the number of molecules per backward pass
    # ndata = len(data_all)
    # batch_size = min(batch_size, ndata) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    shuffle = train_config.get('shuffle', False)
    num_workers = train_config.get('num_workers', 0)
    pin_memory = train_config.get('pin_memory', True)

    # the learning rate scheduler params, by default turned off
    cyclic_range = train_config.get('cyclic_range', [-1, -2]) 
    exp_range = train_config.get('exp_range', [-1, -2]) 
    cos_range = train_config.get('cos_range', [-1, -2]) 
    n_cycles = train_config.get('lr_cycles', 1.0)
    steps_per_cycle = (cyclic_range[1] - cyclic_range[0])/(2*n_cycles)
    learning_rate_deacy_rate = train_config.get('learning_rate_decay_rate', 1.0)
    cycle_lr_min = train_config.get('cycle_lr_min', learning_rate)
    cycle_lr_max = train_config.get('cycle_lr_max', learning_rate)
    cosine_t0 = train_config.get('cosine_t0', n_epochs) # number of epochs per restart
    granular_print = train_config.get('granular_print', False) # number of epochs per restart
    conformer_batch = train_config.get('conformer_batch', 1) # number of epochs per restart
    remaining_data_sample = train_config.get('remaining_data_sample', 0) # number of epochs per restart

    if conformer_batch == 1:
        conformer_alpha = train_config.get('conformer_alpha', 0) # number of epochs per restart
    else:
        conformer_alpha = 1. / (conformer_batch + 1)


    do_conformers = conformer_alpha > 0

    cyclic_scheduler = CyclicLR(optimizer, base_lr=cycle_lr_min, max_lr=cycle_lr_max, step_size_up = steps_per_cycle, step_size_down = steps_per_cycle, cycle_momentum=False)
    exponential_scheduler = ExponentialLR(optimizer, gamma=learning_rate_deacy_rate)
    cos_scheduler = CosineAnnealingWarmRestarts(optimizer, cosine_t0)

    loss_kwargs['rank'] = rank
    loss_kwargs['device'] = device
    loss_kwargs['batch_size'] = batch_size

    data_refine = GraphDataset(data_root, indices_ = indices_refine, rank = rank)
    loss_kwargs['omega_fit'] = data_refine[0].iomega
    criterion = SigmaLoss(**loss_kwargs)
    criterion = criterion.to(device)

    len_all_files = len(os.listdir(f'{data_root}/raw'))
    indices_remaining = np.array([i for i in range(len_all_files) if not i in indices_refine])
    training_scale = getattr(model, 'scale', 1.0)
    print(f'training_scale: {training_scale}')
    for epoch in range(n_epochs):
        t0 = time.time()
        indices_  = np.random.choice(indices_remaining, remaining_data_sample, replace = False)
        data_rest = GraphDataset(data_root, indices_ = indices_, rank = rank)

        loader = DataLoader(data_refine+data_rest, batch_size = batch_size, shuffle = shuffle, pin_memory=pin_memory, num_workers = num_workers) #, worker_init_fn = seed
        for b, batch in enumerate(loader):
            optimizer.zero_grad()   
            loss = 0
            batch = batch.to(device)
            
            for i in range(len(batch)):
                mol = batch[i]
                sigma_ml = model(mol, train = True)
                sigma_true = construct_symmetric_tensor(mol.sigma_ii, mol.sigma_ij, mol, device = device)/training_scale
                loss += criterion(sigma_ml, sigma_true, C_lo_mo = mol.get('C_lo_mo', None), homo_ind = mol.get('homo_ind', None), lumo_ind = mol.get('lumo_ind', None), nmo = mol.nmo, mo_energy = mol.get('mo_energy', None), energy_scale=training_scale)

            loss.backward()
            optimizer.step()    
        
        # print info
        if rank == 0:
            t1 = time.time()
            t = t1-t0
            if loss != 0:
                loss_val = loss.detach().cpu().numpy()
            else:
                loss_val = -1 # means the loss was not computed in the final batch
            step_size = optimizer.param_groups[0]['lr']
            print(f'time for Adam #{epoch}: {t:0.2f}s, main loop value (final batch loss): {loss_val:0.6e}, step_size: {step_size:0.6e}', flush = True) # , big_loss_count: {big_loss_count}

        # learning rate scheduling
        if cyclic_range[0] <= epoch < cyclic_range[1]:
            cyclic_scheduler.step()
        if exp_range[0] <= epoch < exp_range[1]:
            exponential_scheduler.step()
        if cos_range[0] <= epoch < cos_range[1]:
            cos_scheduler.step()

    return model

def train_graph_model_memory_safe(data, model, train_config, seed = 42, loss_kwargs = {}, rank = 0, log_dir = None): 
    """more memory-conservative training function for GNN

    Args:
        data (GraphDataset object): training data object
        model (GNN): initialized GNN model
        train_config (dict): configuration variables for training
        seed (int, optional): random seed for setup_seed(). Defaults to 42.
        loss_kwargs (dict, optional): loss keyword argmuents beta_i and FMO definition. Defaults to {}.
        rank (int, optional): MPI rank for printing/debugging and training multiple ensembles. Defaults to 0.
        log_dir (str, optional): unused, if you want to use tensorboard logging, the SummaryWriter can dump here. Defaults to None.

    Returns:
        trained GNN
    """    
    setup_seed(seed)
    torch_geometric.seed.seed_everything(seed)
    # assert len(data_x) == len(data_y)
    device =  train_config.get('device', None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('device cuda requested but cuda not available, using CPU')
        device = 'cpu'

    if rank == 0:
        print('train_config: ', train_config, flush = True)
        print('loss_kwargs: ', loss_kwargs, flush = True)
        print('Device used for training: ', device, flush = True)
        total_num_params = count_parameters(model)
        print('Model total number of params: ', total_num_params, flush = True)

    model.to(device)
    freqs_dos = np.linspace(-1, 1, 201)
    
    n_epochs = int(train_config.get('epochs', None))
    if n_epochs is None:
        n_epochs = int(train_config.get('steps', 1000))
        
    weight_decay = train_config.get('weight_decay', 0.0)
    learning_rate = train_config.get('learning_rate', 0.001)
    batch_size = train_config.get('batch_size', np.inf) # the number of molecules per backward pass
    ndata = len(data)
    batch_size = min(batch_size, ndata) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    shuffle = train_config.get('shuffle', False)
    num_workers = train_config.get('num_workers', 0)
    pin_memory = train_config.get('pin_memory', True)

    # the learning rate scheduler params, by default turned off
    cyclic_range = train_config.get('cyclic_range', [-1, -2]) 
    exp_range = train_config.get('exp_range', [-1, -2]) 
    cos_range = train_config.get('cos_range', [-1, -2]) 
    n_cycles = train_config.get('lr_cycles', 1.0)
    steps_per_cycle = (cyclic_range[1] - cyclic_range[0])/(2*n_cycles)
    learning_rate_deacy_rate = train_config.get('learning_rate_decay_rate', 1.0)
    cycle_lr_min = train_config.get('cycle_lr_min', learning_rate)
    cycle_lr_max = train_config.get('cycle_lr_max', learning_rate)
    cosine_t0 = train_config.get('cosine_t0', n_epochs) # number of epochs per restart
    granular_print = train_config.get('granular_print', False) # number of epochs per restart
    conformer_batch = train_config.get('conformer_batch', 1) # number of epochs per restart

    if conformer_batch == 1:
        conformer_alpha = train_config.get('conformer_alpha', 0) # number of epochs per restart
    else:
        conformer_alpha = 1. / (conformer_batch + 1)


    do_conformers = conformer_alpha > 0

    cyclic_scheduler = CyclicLR(optimizer, base_lr=cycle_lr_min, max_lr=cycle_lr_max, step_size_up = steps_per_cycle, step_size_down = steps_per_cycle, cycle_momentum=False)
    exponential_scheduler = ExponentialLR(optimizer, gamma=learning_rate_deacy_rate)
    cos_scheduler = CosineAnnealingWarmRestarts(optimizer, cosine_t0)

    loss_kwargs['rank'] = rank
    loss_kwargs['device'] = device
    loss_kwargs['batch_size'] = batch_size
    loss_kwargs['omega_fit'] = data[0].iomega
    
    criterion = SigmaLoss(**loss_kwargs)
    criterion = criterion.to(device)

    # if not log_dir is None and rank == 0:
    #     writer = SummaryWriter(log_dir=log_dir)

    loader = DataLoader(data, batch_size = batch_size, shuffle = shuffle, pin_memory=pin_memory, num_workers = num_workers) #, worker_init_fn = seed
    training_scale = getattr(model, 'scale', 1.0)
    print(f'training_scale: {training_scale}', flush = True)
    for epoch in range(n_epochs):
        t0 = time.time()
        for b, batch in enumerate(loader):
            optimizer.zero_grad()   
            loss = 0
            
            batch.x = batch.x.to(device) 
            batch.edge_attr = batch.edge_attr.to(device) 
            batch.edge_index = batch.edge_index.to(device) 
            
            for i in range(len(batch)):
                mol = batch[i]
                sigma_ml = model(mol, train = True)
                sigma_true = construct_symmetric_tensor(mol.sigma_ii, mol.sigma_ij, mol, device = device)/training_scale
                C_lo_mo = mol.get('C_lo_mo', None)
                if not C_lo_mo is None:
                    C_lo_mo = C_lo_mo.to(device) 

                loss += criterion(sigma_ml, sigma_true, C_lo_mo = C_lo_mo, homo_ind = mol.get('homo_ind', None), lumo_ind = mol.get('lumo_ind', None), nmo = mol.nmo, mo_energy = mol.get('mo_energy', None), energy_scale = training_scale)

            loss.backward()
            optimizer.step()  
              
        
        # write log
        if rank == 0:
            t1 = time.time()
            t = t1-t0
            if loss != 0:
                loss_val = loss.detach().cpu().numpy()
            else:
                loss_val = -1 # means the loss was not computed in the final batch
            step_size = optimizer.param_groups[0]['lr']
            print(f'time for Adam #{epoch}: {t:0.2f}s, main loop value (final batch loss): {loss_val:0.6e}, step_size: {step_size:0.6e}', flush = True)
            
            # write a log file for tensorboard
            # if not log_dir is None: 
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             writer.add_histogram(name + '/values', param.data, global_step=epoch)
            #             writer.add_histogram(name + '/grad', param.grad, global_step=epoch)  

        # learning rate scheduling
        if cyclic_range[0] <= epoch < cyclic_range[1]:
            cyclic_scheduler.step()
        if exp_range[0] <= epoch < exp_range[1]:
            exponential_scheduler.step()
        if cos_range[0] <= epoch < cos_range[1]:
            cos_scheduler.step()

    return model