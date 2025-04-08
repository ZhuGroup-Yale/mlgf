import torch
import torch.nn.functional as F
from torch import Tensor


from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.conv import MessagePassing
from mlgf.model.pytorch.data import construct_symmetric_tensor
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.data import Data as Graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d#, Linear


import os
import numpy as np
import random
import argparse
import joblib
import time
import warnings
import psutil
import copy

from mlgf.model.pytorch.base import TorchBase

class EdgeUpdate(TorchBase):
    

    def __init__(self, **kwargs):
        
        super(EdgeUpdate, self).__init__(**kwargs)
        """u_e in paper
        """        
        
        self.n_input, self.n_hidden, self.n_output = kwargs.get('n_input', 30),  kwargs.get('n_hidden', 60),  kwargs.get('n_output', 20)

        self.Lin = nn.Linear(self.n_input, self.n_output)
        self.act = nn.SiLU()
        
    def forward(self, x): 
        return self.act(self.Lin(x))

class MessagePassingUpdate(TorchBase):

    def __init__(self, **kwargs):
        
        super(MessagePassingUpdate, self).__init__(**kwargs)
        """u_x in paper
        """        
        
        self.n_input = kwargs.get('n_input', 20)
        self.n_output = kwargs.get('n_output', 20)

        self.Lin = nn.Linear(self.n_input, self.n_output, bias = True)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.Lin(x))

# second argument to AttentionalAggregation
class AttentionMessage(TorchBase):

    def __init__(self, **kwargs):
        
        super(AttentionMessage, self).__init__(**kwargs)
        """m^l_{ij} in paper
        """        
        
        self.node_attr = kwargs.get('node_attr', 20)
        self.edge_start = 2*self.node_attr
        self.n_edge_attr = kwargs.get('edge_attr', 20)
        self.n_output = kwargs.get('channels', 20)

        self.Lin1_x1 = nn.Linear(self.node_attr, self.n_edge_attr, bias = False)
        self.Lin1_x2 = nn.Linear(self.node_attr, self.n_edge_attr, bias = False)
        self.Lin2 = nn.Linear(self.n_edge_attr*3, self.n_output, bias = True)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act_fin = nn.SiLU()
        

    def forward(self, x):
        x_edge = x[:, self.edge_start:]
        x1 = self.act1(self.Lin1_x1(x[:, :self.node_attr] + x[:, self.node_attr:self.edge_start]))
        x2 = self.act2(self.Lin1_x2(torch.abs(x[:, :self.node_attr] - x[:, self.node_attr:self.edge_start])))
        return self.act_fin(self.Lin2(torch.cat((x1, x2, x_edge), dim = 1)))

# first argument to AttentionalAggregation
class AttentionScore(TorchBase):

    def __init__(self, **kwargs):
        
        super(AttentionScore, self).__init__(**kwargs)
        """s^l_{ij} in paper
        """       
        
        self.n_input = kwargs.get('node_attr', 20)
        self.edge_attr = kwargs.get('edge_attr', 20)
        self.n2_input = 2*self.n_input

        self.Lin_x1 = nn.Linear(self.n_input, self.edge_attr, bias = False)
        self.Lin_x2 = nn.Linear(self.n_input, self.edge_attr, bias = False)
        self.Lin_all = nn.Linear(self.edge_attr*3, 1, bias = False)
        self.act = nn.Tanhshrink()

    def forward(self, x):
        x_edge = x[:, self.n2_input:]
        x1 = self.Lin_x1(x[:, :self.n_input] + x[:, self.n_input:self.n2_input])
        x2 = self.Lin_x2(torch.abs(x[:, :self.n_input] - x[:, self.n_input:self.n2_input]))
        x = self.Lin_all(torch.cat((x1, x2, x_edge), dim = 1))
        x = self.act(x)
        return x

class EdgeDecoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(EdgeDecoder, self).__init__(**kwargs)
        """decodes into \Sigma_{ij}(iomega) cat(real, imag)
        """        

        self.n_input = kwargs.get('n_input', 10)
        self.n_hidden = kwargs.get('n_hidden', 60)
        self.n_output = kwargs.get('n_output', 36)

        self.Lin1_re = nn.Linear(self.n_input, self.n_hidden, bias = True)
        self.Lin1_im = nn.Linear(self.n_input, self.n_hidden, bias = True)
        self.Lin3_re = nn.Linear(self.n_hidden, self.n_output//2, bias = True)
        self.Lin3_im = nn.Linear(self.n_hidden, self.n_output//2, bias = True)

        self.a1_re = nn.SiLU()
        self.a1_im = nn.SiLU()
        
    def forward(self, x):
        x_re = self.a1_re(self.Lin1_re(x))
        x_im = self.a1_im(self.Lin1_im(x))
        return torch.cat([self.Lin3_re(x_re), self.Lin3_im(x_im)], dim = -1)

class NodeDecoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(NodeDecoder, self).__init__(**kwargs)
        """decodes into \Sigma_{ii}(iomega) cat(real, imag)
        """        
        
        self.n_input = kwargs.get('n_input', 10)
        self.n_hidden = kwargs.get('n_hidden', 60)
        self.n_output = kwargs.get('n_output', 36)

        self.Lin1_re = nn.Linear(self.n_input, self.n_hidden, bias = True)
        self.Lin1_im = nn.Linear(self.n_input, self.n_hidden, bias = True)
        self.Lin3_re = nn.Linear(self.n_hidden, self.n_output//2, bias = True)
        self.Lin3_im = nn.Linear(self.n_hidden, self.n_output//2, bias = True)

        self.a1_re = nn.SiLU()
        self.a1_im = nn.SiLU()
        
    def forward(self, x):
        x_re = self.a1_re(self.Lin1_re(x))
        x_im = self.a1_im(self.Lin1_im(x))
        return torch.cat([self.Lin3_re(x_re), self.Lin3_im(x_im)], dim = -1)


class NodeEncoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(NodeEncoder, self).__init__(**kwargs)
        """returns x^0_i in paper
        """        
        
        self.nstatic, self.ndynamical, self.ncat = kwargs.get('nstatic', 6), kwargs.get('ndynamical', 24), kwargs.get('ncat', 14)

        self.nstatic_layer = kwargs.get('nstatic_layer', 150) #20
        self.ndynamical_layer = kwargs.get('ndynamical_layer', 350) #200
        self.cat_layer_width =  kwargs.get('ncat_layer', 10*self.ncat) #2*

        self.hidden_layer_width = self.cat_layer_width+self.nstatic_layer+self.ndynamical_layer
        self.encoding_width = kwargs.get('encoding_width', 10)

        self.Lin1_static = nn.Linear(self.nstatic, self.nstatic_layer)
        self.Lin1_dyn = nn.Linear(self.ndynamical, self.ndynamical_layer)
        self.Lin1_cat = nn.Linear(self.ncat, self.cat_layer_width)
        self.Lin2 = nn.Linear(self.hidden_layer_width, self.hidden_layer_width, bias = True)
        self.a1s = nn.SiLU()
        self.a1d = nn.SiLU()
        self.a1c = nn.SiLU()
        self.a1sum = nn.SiLU()
        self.a2 = nn.SiLU()
        self.Lin3 = nn.Linear(self.hidden_layer_width, self.encoding_width, bias = True)
        self.shortcut = nn.Linear(self.nstatic + self.ndynamical, self.encoding_width, bias = True)
        
    def forward(self, x): # x_static, x_dynamical, x_categorical
        xs, xd = x[:,:self.nstatic], x[:,self.nstatic:(self.nstatic+self.ndynamical)]
        xc = x[:, (self.nstatic+self.ndynamical):] # categorical features, should already be one hot encoded to avoid cost
        xs = self.a1s(self.Lin1_static(xs))
        xd = self.a1d(self.Lin1_dyn(xd))
        xc = self.a1c(self.Lin1_cat(xc))
        xt = torch.cat((xs, xd, xc), dim = 1)
        xt = self.Lin2(xt)
        xt = self.Lin3(self.a1sum(xt)) + self.shortcut(x[:,:(self.nstatic+self.ndynamical)])
        return self.a2(xt)

class EdgeEncoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(EdgeEncoder, self).__init__(**kwargs)
        """returns e^0_{ij} in paper
        """        
        
        self.nstatic, self.ndynamical, self.ncat = kwargs.get('nstatic', 6), kwargs.get('ndynamical', 24), kwargs.get('ncat', 14)

        self.nstatic_layer = kwargs.get('nstatic_layer', 150) #20
        self.ndynamical_layer = kwargs.get('ndynamical_layer', 350) #200

        self.hidden_layer_width = self.nstatic_layer+self.ndynamical_layer
        self.encoding_width = kwargs.get('encoding_width', 10)

        self.Lin1_static = nn.Linear(self.nstatic, self.nstatic_layer)
        self.Lin1_dyn = nn.Linear(self.ndynamical, self.ndynamical_layer)
        self.Lin2 = nn.Linear(self.hidden_layer_width, self.hidden_layer_width, bias = True)
        self.a1s = nn.SiLU()
        self.a1d = nn.SiLU()
        self.a1sum = nn.SiLU()
        self.a2 = nn.SiLU()
        self.Lin3 = nn.Linear(self.hidden_layer_width, self.encoding_width, bias = True)
        self.shortcut = nn.Linear(self.nstatic + self.ndynamical, self.encoding_width, bias = True)

        
    def forward(self, x): # x_static, x_dynamical, x_categorical
        xs, xd = x[:,:self.nstatic], x[:,self.nstatic:(self.nstatic+self.ndynamical)]
        xs = self.a1s(self.Lin1_static(xs))
        xd = self.a1d(self.Lin1_dyn(xd))
        xt = torch.cat((xs, xd), dim = 1)
        xt = self.Lin2(xt)
        xt = self.Lin3(self.a1sum(xt)) + self.shortcut(x)
        return self.a2(xt)


class NodeMPL(MessagePassing):
    def __init__(self, channels,
                 aggr = 'add', **kwargs):  
        super().__init__(aggr=aggr, **kwargs)
        
        """Single message passing layer (core of the GNN implementation)

        Args:
            channels (int): N_c in paper
            aggr (str, optional OR torch_geometric aggr object): the aggregation mechanism. Defaults to 'add'.
        """      

        self.mlp = MessagePassingUpdate(n_input = channels, n_output = channels)

    def message(self, x_i, x_j, edge_attr)-> Tensor:
        return torch.cat((x_i, x_j, edge_attr), dim = -1)

    def forward(self, x, edge_index, edge_attr):

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        node_features = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return self.mlp(node_features)

class GNN(TorchBase):
    def __init__(self, **kwargs):
        super(GNN, self).__init__(**kwargs)
        """MBGF-Net
        """        

        self.n_out = kwargs.get('n_out', 36)
        self.scale = kwargs.get('scale_y', 0.001) 

        self.node_nstatic = kwargs.get('nstatic_ii', 6)
        self.node_ndynamical = kwargs.get('ndynamical_ii', 24)
        self.ncat_ii = kwargs.get('ncat_ii', 3)
        self.edge_nstatic = kwargs.get('nstatic_ij', 6)
        self.edge_ndynamical = kwargs.get('ndynamical_ij', 24)

        self.node_nstatic_layer = kwargs.get('node_nstatic_layer', 100)
        self.node_ndynamical_layer = kwargs.get('node_ndynamical_layer', 300) 
        self.node_ncat_layer = kwargs.get('node_ncat_layer', 20)   

        self.edge_nstatic_layer = kwargs.get('edge_nstatic_layer', 30)
        self.edge_ndynamical_layer = kwargs.get('edge_ndynamical_layer', 60)   

        self.edge_decoder_nhidden =  kwargs.get('edge_decoder_nhidden', 60)
        self.node_decoder_nhidden =  kwargs.get('node_decoder_nhidden', 60)
        self.mpl_channels =  kwargs.get('mpl_channels', 48)
        self.message_passing_updates =  kwargs.get('message_passing_updates', 3)
        self.decoder_width = (self.message_passing_updates + 1)*self.mpl_channels

        self.NodeEncoder = NodeEncoder(encoding_width = self.mpl_channels, nstatic = self.node_nstatic, ndynamical = self.node_ndynamical, ncat = self.ncat_ii, nstatic_layer = self.node_nstatic_layer, ndynamical_layer = self.node_ndynamical_layer, ncat_layer = self.node_ncat_layer)
        self.EdgeEncoder = EdgeEncoder(encoding_width = self.mpl_channels, nstatic = self.edge_nstatic, ndynamical = self.edge_ndynamical, nstatic_layer = self.edge_nstatic_layer, ndynamical_layer = self.edge_ndynamical_layer) 

        self.aggr_msgs = nn.ModuleList([AttentionMessage(node_attr = self.mpl_channels, edge_attr = self.mpl_channels, channels = self.mpl_channels) for i in range(self.message_passing_updates)])
        self.aggr_scores = nn.ModuleList([AttentionScore(node_attr = self.mpl_channels, edge_attr = self.mpl_channels) for i in range(self.message_passing_updates)])
        self.aggrs = nn.ModuleList([AttentionalAggregation(self.aggr_scores[i], self.aggr_msgs[i]) for i in range(self.message_passing_updates)])
        self.convs = nn.ModuleList([NodeMPL(channels = self.mpl_channels, aggr = self.aggrs[i]) for i in range(self.message_passing_updates)])
        self.edge_updates = nn.ModuleList([EdgeUpdate(n_input = self.mpl_channels, n_output = self.mpl_channels) for i in range(self.message_passing_updates)])

        self.node_decoder = NodeDecoder(n_input = self.decoder_width, n_hidden = self.decoder_width, n_output = self.n_out)
        self.edge_decoder = NodeDecoder(n_input = self.decoder_width, n_hidden = self.decoder_width, n_output = self.n_out)
        self.double() # convert all model_parameters to float64 to agree with numpy arrays converted to tensors in GraphOrchestrator


    def forward(self, graph, **kwargs): 
        train = kwargs.get('train', False) # if in train mode, doesn't scale the self-energy by self.scale. Sometimes, learning is easier in non-a.u.
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x_hiddens = []
        e_hiddens = []

        x_hiddens.append(self.NodeEncoder(x))
        e_hiddens.append(self.EdgeEncoder(edge_attr))

        for i in range(self.message_passing_updates):
            e_new = self.edge_updates[i](self.aggr_msgs[i](self.get_er_feature(x_hiddens[-1], edge_index, e_hiddens[-1])))
            x_new = self.convs[i](x_hiddens[-1], edge_index, e_hiddens[-1])
            x_hiddens.append(x_new.clone())
            e_hiddens.append(e_new.clone())

        sigma_ii = self.node_decoder(torch.cat(x_hiddens, dim = -1))
        sigma_ij = self.edge_decoder(torch.cat(e_hiddens, dim = -1))

        if train:
            return construct_symmetric_tensor(sigma_ii, sigma_ij, graph)
        else:
            return construct_symmetric_tensor(sigma_ii, sigma_ij, graph)*self.scale

    def get_er_feature(self, x, edge_index, edge_attr):
        """returns all features associated with edge (x_i, x_j, e_ij)

        Args:
            x (torch.float64): node attributes
            edge_index (torch.int): indices of edges (2 x n_edges)
            edge_attr (torch.float64): edge attributes

        Returns:
            torch.float64: concatenation of x_i, x_j, edge_attr in order of edge_index
        """        
        x_i, x_j = x[edge_index[0, :], :], x[edge_index[1, :], :]
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return x