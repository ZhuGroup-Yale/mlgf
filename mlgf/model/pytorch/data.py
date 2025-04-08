import os

import torch
from torch_geometric.data import Dataset as torchDataset
from torch_geometric.data import Data as torchData
# torch.set_default_dtype(torch.float64)

class GraphDataset(torchDataset):
    """A collection of DFT graph objects

    Args:
        torchDataset (Dataset from torch_geometric): inherits most functionality from torch_geometric
    """    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, in_memory = False, indices_ = None, rank = None):
        
        self.rank = rank
        super().__init__(root, transform, pre_transform, pre_filter)
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.in_memory = in_memory
        
        if self.in_memory:
            self.data = self.load_in_memory()
        
        else:
            self.data = None
        
        # print('pytorch dset raw_dir: ', self.raw_dir)
        if indices_ is None:
            length = len(os.listdir(f'{self.raw_dir}'))
            self.indices_ = list(range(length))
        else:
            self.indices_ = indices_
        if not rank is None:
            self.processed_dir = f'{root}/processed_{rank}'
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return []

    @property
    def processed_dir(self):
        if not self.rank is None:
            self.processed_dir = f'{self.root}/processed_{self.rank}'
        else:
            self.processed_dir = f'{self.root}/processed'

    @processed_dir.setter
    def processed_dir(self, rank):
        return f'{self.root}/processed_{rank}'
    
    def len(self):
        return len(self.indices_)

    def download(self):
        pass

    def load_in_memory(self):
        return [torch.load(os.path.join(self.raw_dir, f'data_{idx}.pt'))  for idx in range(self.len())]
    
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.in_memory:
            return self.data[idx]
        else:
            return torch.load(os.path.join(self.raw_dir, 
                                 f'data_{self.indices_[idx]}.pt'))   

class Graph(torchData):
    """A DFT graph object

    Args:
        torchData (Data from torch_geometric): inherits most functionality from torch_geometric
    """    
    def __init__(self, x = None, edge_attr=None, edge_index=None):
        super().__init__(x = x, edge_attr=edge_attr, edge_index=edge_index)
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """

def construct_symmetric_tensor(tensor1, tensor2, graph, device = None):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        graph (gnn_orchestrator.GraphData object): the graph object that stores the indices of nodes and edges that have not been removed from the DFT graph

    Returns:
        torch.float64: cat[sigma(iw).real, sigma(iw).imag]
    """    
    edge_indices, node_indices = graph.edge_indices_nonzero, graph.node_indices_nonzero
    # Initialize the rank-3 tensor with zeros
    nmo = graph.nmo
    nw = tensor1.size(-1)

    result_tensor = torch.zeros(nmo, nmo, nw, dtype=tensor1.dtype, device=tensor1.device)

    # Fill the diagonal with tensor1 entries
    result_tensor[node_indices, node_indices, :] = tensor1

    # Fill the upper and lower triangle with tensor2 entries (graph is undirected so we don't have to flip stuff, though it could save computation)
    result_tensor[edge_indices[:,0], edge_indices[:,1], :] = tensor2

    # saves memory if tensor1 and tensor2 are not loaded onto GPU before this function call
    if not device is None:
        result_tensor = result_tensor.to(device)

    return result_tensor

def unravel_rank2(arr):
    """unravels a matrix for data processing in GraphDataset (e.g. an orbital rotation matrix)

    Args:
        arr (torch.float64): 2d matrix, Norb x Norb

    Returns:
        torch.float64: 1d vector, (Norb * Norb) x 1
    """    
    # Unravel the N x N array into an N^2 x 1 array
    unraveled_array = arr.flatten()

    return unraveled_array

def reconstruct_rank2(unraveled_array, nmo):
    """inverts unravel_rank2, i.e. get back 2d matrix from 1d

    Args:
        unraveled_array (torch.float64): 1d
        nmo (int): number of nmo

    Returns:
        torch.float64: 2d, Norb x Norb
    """    
    # Reconstruct the N^2 x 1 array into an N x N array
    reconstructed_array = unraveled_array.view(nmo, nmo)

    return reconstructed_array