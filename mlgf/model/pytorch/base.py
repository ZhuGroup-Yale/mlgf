import torch
import torch.nn as nn
import torch.nn.functional as F

# a wrapper around the nn.Module class so arbitrary **kwargs can be passed to each of the following models
# you can just add new models here of any name, just make sure you create an alias and add it to the pt_alias.py function
class TorchBase(nn.Module):
    def __init__(self, **kwargs): 
        super(TorchBase, self).__init__()
        pass
