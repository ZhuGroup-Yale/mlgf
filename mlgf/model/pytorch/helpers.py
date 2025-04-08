from mlgf.model.pytorch.pt_alias import get_model_from_alias
from mlgf.model.pytorch.loss import SigmaLoss
from mlgf.model.pytorch.data import construct_symmetric_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from torch.func import stack_module_state
from torch.func import functional_call
from torch import vmap

# torch.set_default_dtype(torch.float64)

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

def count_parameters(model):
    """A function to that tallies all the parameters in the model by recursively getting submodules of model
    Args:
        model (_type_): GNN

    Returns:
        int: parameter count
    """    
    def get_num_params(module, prefix=''):
        total_params = 0
        for name, submodule in module.named_children():
            submodule_params = sum(p.numel() for p in submodule.parameters())
            submodule_prefix = f"{prefix}.{name}" if prefix else name
            # print(f"{submodule_prefix}: {submodule_params} params")
            total_params += submodule_params

            # Recursively count params for submodules, handling ModuleList and Sequential
            if isinstance(submodule, (nn.ModuleList, nn.Sequential)):
                for idx, sm in enumerate(submodule):
                    sm_prefix = f"{submodule_prefix}[{idx}]"
                    total_params += get_num_params(sm, sm_prefix)
            else:
                total_params += get_num_params(submodule, submodule_prefix)
                
        return total_params

    tot_params = get_num_params(model)
    return tot_params

def get_one_hot_torch(cat_feature, num_classes = -1):
    """convert an N_datax1 categorical feature to an Nxd vector of 0s and 1s where d is the number of categories

    Args:
        cat_feature (np.int): N_data x 1
        num_classes (int, optional): number of classes in the category. Defaults to -1.

    Returns:
        torch.int: binary data
    """    
    return F.one_hot(torch.from_numpy(cat_feature), num_classes=num_classes)

def setup_seed(seed):
    """randomizes everything to start a NN optimization (initial weights, dropout sequence, minibatch selection, etc)

    Args:
        seed (int): e.g. 42
    """    
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

def one_hot_encode_category(categories, num_classes_list):
    """One hot encode an integer array

    Args:
        categories (np.int): integer array, N_data x N_categories
        num_classes_list (np.int): the number of categories for each column, 1 x N_categories

    Returns:
        torch.float64: the binary data
    """    
    for i in range(categories.shape[-1]):
        new_one_hot = get_one_hot_torch(categories[:, i], num_classes = num_classes_list[i])
        if i == 0:
            one_hot = new_one_hot
        else:
            one_hot = torch.cat((one_hot, new_one_hot), dim = 1)
    return one_hot

def get_graph_ensemble_mean(arr):
    """get ensemble mean for self-energy, works for any arbitrary ensemble tensor retured by predict_wrapper_graph_ensemble

    Args:
        ar (torch.float64): N_models x N_orb x N_orb x nomega
    Returns:
        torch.float64:  mean, N_orb x N_orb x nomega
    """    
    return np.mean(arr, axis = 0)

def get_graph_ensemble_uncertainty(values, values_mean = None):
    """get uncertainty estimate for self-energy
    Calculates a simple, purely epistemic uncertainty with a formula this paper:
    https://pubs.rsc.org/en/content/articlelanding/2017/SC/C7SC02267K
    works for any arbitrary ensemble tensor retured by predict_wrapper_graph_ensemble of shape N_ensemble_members x shape1 x shape2 x ... etc

    Args:
        values (torch.float64): N_models x N_orb x N_orb x nomega
        values_mean (torch.float64, optional): N_orb x N_orb x nomega. Defaults to None.

    Returns:
        torch.float64:  uncertainty, N_orb x N_orb x nomega
    """    
    if values_mean is None:
        values_mean = get_graph_ensemble_mean(values)          
    return (np.sum((values-values_mean)**2, axis = 0)/(values.shape[0] - 1 ))**0.5

def load_pt_model(params, model_name, **kwargs):
    """load a torch model

    Args:
        params (OrderedDict): parameters for GNN (from previous training)
        model_name (str): name of model for pt_alias.py

    Returns:
        GNN
    """    
    model = get_model_from_alias(model_name, **kwargs)
    model.load_state_dict(params, strict = kwargs.get('load_strict', True))
    model.eval()
    return model

def predict_wrapper_graph_ensemble(graph, model_states, model_alias, **kwargs):
    """predicts self-energy from list of GNN model states 

    Args:
        graph (GraphData object): graph to predict self-energy for
        model_states (list[OrderedDict]): list of parameters for GNN ensemble (from previous training)
        model_alias (str): name of model for pt_alias.py

    Returns:
        torch.float64: sigma(iw)
    """    
    sigmas = []
    for i in range(len(model_states)):
        sigma = predict_wrapper_graph(graph, model_states[i], model_alias, **kwargs)
        sigmas.append(sigma)
    
    return torch.stack(sigmas).numpy()

def predict_wrapper_graph(graph, model_state, model_alias, **kwargs):
    """predicts self-energy from torch 

    Args:
        graph (GraphData object): graph to predict self-energy for
        model_state (OrderedDict): parameters for GNN (from previous training)
        model_alias (str): name of model for pt_alias.py

    Returns:
        torch.float64: sigma(iw)
    """    
    
    model = load_pt_model(model_state, model_alias, **kwargs) 
    model.to(kwargs.get('device', 'cpu'))
    sigma = model(graph).detach().cpu()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return sigma