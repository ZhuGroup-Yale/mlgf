import torch
import torch.nn.functional as F
from torch import Tensor
from mlgf.model.pytorch.base import TorchBase


import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import random
import argparse
import joblib
import time
import warnings
import psutil

from mlgf.model.pytorch.data import reconstruct_rank2

class SigmaLoss(TorchBase):
    def __init__(self, **kwargs):
        super(SigmaLoss, self).__init__(**kwargs)
        """Self-energy loss function
        smoothness term not used (2nd derivative)
        """        
        self.rank = kwargs.get('rank', 0)
        self.smoothness_weight = kwargs.get('smoothness_weight', 0.0)
        self.frontier_weight = kwargs.get('frontier_weight', 0.0)
        self.gradient_weight = kwargs.get('gradient_weight', 0.0)
        self.frontier_range = kwargs.get('frontier_range', None) # first entry A is homo - A, second B is lumo + B
        self.x_values = kwargs.get('omega_fit', None)
        self.energy_scale = kwargs.get('energy_scale', 1.0)
        self.debug = kwargs.get('debug', False)
        if self.x_values is None:
            self.x_values = torch.tensor([7.157855215121831e-05, 0.004063370586927713 ,0.017253364861917506 ,0.03590146217196803 ,0.06254742321027994 ,0.0987009331103681 ,0.14658084310194658 ,0.19545991145782668, 
            0.27362424796275964, 0.35391256893386824,0.4846115218635232,0.6225728336689219,0.8025963694701741,1.0427064170951983,1.3718937669882554,1.8391918614554528,2.532904118752841, 3.6253805205356855])
        # self.x_values = self.x_values*self.energy_scale
        if self.rank == 0:
            print(self.x_values)
         # finite difference dx values for smoothness penalty, from https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
        self.x1_d2 = self.x_values[2:] - self.x_values[1:-1]
        self.x2_d2 = self.x_values[1:-1] - self.x_values[:-2]
        self.x3_d2 = self.x_values[2:] - self.x_values[:-2] 

         # finite difference dx values for gradient penalty
        self.x_d1 = self.x_values[1:] - self.x_values[:-1]
        self.nw = len(self.x_values)
        self.device = kwargs.get('device', 'cpu')
        self.first_n_grad_re = kwargs.get('first_n_grad_re', self.nw-1) 
        self.first_n_grad_im = kwargs.get('first_n_grad_im', self.nw-1) 

        print('-------Loss function initialization of precision and device-------')
        for attr_name, attr_value in vars(self).items():
            if attr_name == 'device':
                continue
            try:
                self.__setattr__(attr_name, attr_value.to(self.device))
                if self.rank == 0:
                    print(f'attr {attr_name} put on device: {self.device}')
            except AttributeError:
                pass
                # print(f'skpping attr {attr_name} ( not moved to {self.device} )')

        for attr_name, attr_value in vars(self).items():
            if attr_name == 'device':
                continue
            try:
                self.__setattr__(attr_name, self.convert_precision(attr_value))
                if self.rank == 0:
                    print(f'attr {attr_name} switched to precision: {self.precision}')
            except AttributeError:
                pass
                # print(f'skpping precision change of attr {attr_name} ( precision cannot be changed )')
        if self.rank == 0:
            print('-------End Loss function initialization-------')

    # sigma_predicted real or im
    def smoothness_sigma(self, sigma):
        """2nd deriative

        Args:
            sigma (torch.float64)

        Returns:
            2nd deriative
        """        
        return ((sigma[:, :, 2:] - sigma[:, :, 1:-1]) / (self.x1_d2) -
             (sigma[:, :, 1:-1] - sigma[:, :, :-2]) / (self.x2_d2)) /(self.x3_d2)
    
    # sigma real or im
    def gradient_sigma(self, sigma):
        """1st deriative via FD

        Args:
            sigma (torch.float64)

        Returns:
            torch.float64, 2nd deriative
        """        
        return (sigma[:, :, 1:] - sigma[:, :, :-1]) / self.x_d1

    def mo_penalty(self, diff, C_lo_mo, frontiers):
        """MO penalty in paper

        Args:
            diff (torch.float64): sigma_true minus sigma_pred
            C_lo_mo (torch.float64): Norb x Norb SAIAO to MO rotation matrix
            frontiers (list[int]): list of frontier orbital indices

        Returns:
            FMO penalty
        """        
        mo_diff = torch.einsum('ji,jkn,kl->iln', C_lo_mo, diff, C_lo_mo)
        return torch.mean(mo_diff[frontiers, frontiers, :]**2) # or sum

    # def remove_core_mo(self, frontier_range, mo_energy, homo_ind, below_homo_thresh = -2):
    #     """remove core orbitals from FMO list

    #     Args:
    #         frontier_range (list[int]): original list of FMO
    #         mo_energy (torch.float64): KS orbital energies from DFT
    #         homo_ind (int): index of HOMO
    #         below_homo_thresh (int, optional): Remove FMOs >2 a.u. below the HOMO. Defaults to -2.

    #     Returns:
    #         list[int]: cleaned indices
    #     """        
    #     homo = mo_energy[homo_ind]
    #     return [f for f in frontier_range if (mo_energy[f] - homo)> below_homo_thresh]

    def get_qpe_range(self, homo_ind, lumo_ind, nmo):
        if self.frontier_range is None:
            frontiers = torch.arange(0, nmo)

        if type(self.frontier_range[0])==int: 
            top_frontier = min(lumo_ind + 1 + self.frontier_range[1], nmo)
            bot_frontier = max(homo_ind - self.frontier_range[0], 0)
            frontiers = torch.arange(bot_frontier, top_frontier)

        if type(self.frontier_range[0])==float:
            nocc = homo_ind + 1
            nvirt = nmo - nocc
        
            lumo_end = int(nvirt*.4)+1
            top_frontier = min(lumo_ind+int(nvirt*self.frontier_range[1])+1, nmo)
            bot_frontier = max(homo_ind - int(nocc*self.frontier_range[0]), 0)
            frontiers = torch.arange(bot_frontier, top_frontier)

            # print(nmo, nocc, nvirt, frontiers, flush = True)
        # frontiers = self.remove_core_mo(frontiers, mo_energy, homo_ind)
        return frontiers
                
    def forward(self, sigma_predicted, sigma_true, C_lo_mo = None, homo_ind = None, lumo_ind = None, nmo = None, mo_energy = None, energy_scale = 1.0): #
        """forward method for self-energy loss function

        Args:
            sigma_predicted (torch.float64): predicted self-energy (Norb x Norb x Nomega)
            sigma_true (torch.float64): true self-energy (Norb x Norb x Nomega)
            C_lo_mo (torch.float64, optional): SAIAO to MO rotation unraveled (Norb*Norb x 1). Defaults to None.
            homo_ind (int, optional): index of HOMO. Defaults to None.
            lumo_ind (int, optional): index of LUMO. Defaults to None.
            nmo (int, optional): Number of orbitals. Defaults to None.
            mo_energy (torch.float64, optional): KS orbital energies from DFT. Defaults to None.
            energy_scale (float64, optional): training scale (0.001 corresponds to m a.u.). Defaults to 1.0.

        Returns:
            torch.float64: loss for single molecule
        """
        diff = sigma_predicted - sigma_true
        loss = torch.mean(diff**2)
        if self.debug:
            print_loss = loss.cpu().detach().numpy()
            print('local orbital (SAIAO) penalty: ', print_loss)

        # FMO term
        if self.frontier_weight > 0:
            C_lo_mo = reconstruct_rank2(C_lo_mo, nmo)
            frontiers = self.get_qpe_range(homo_ind, lumo_ind, nmo)
            mo_penalty = self.frontier_weight*self.mo_penalty(diff, C_lo_mo, frontiers)#/beta
            if self.debug:
                print_loss = mo_penalty.cpu().detach().numpy()
                print('mo penalty: ', print_loss)
            loss += mo_penalty
        
        # unused term 
        if self.smoothness_weight > 0:
            # re, im
            d2dx_re = self.smoothness_sigma(sigma_predicted[:,:,:self.nw])
            d2dx_im = self.smoothness_sigma(sigma_predicted[:,:,self.nw:])
            smoothness_loss = self.smoothness_weight*torch.mean((d2dx_re**2 + d2dx_im**2))#/beta
            if self.debug:
                print_loss = smoothness_loss.cpu().detach().numpy()
                print('smoothness penalty: ', print_loss)
            loss += smoothness_loss

        if self.gradient_weight > 0 and energy_scale != 0:
            if self.first_n_grad_re > 0:
                grad_diff_re = energy_scale*(self.gradient_sigma(sigma_predicted[:,:,:self.nw])-self.gradient_sigma(sigma_true[:,:,:self.nw]))
                loss += (self.gradient_weight)*torch.mean(grad_diff_re[:,:,:self.first_n_grad_re]**2)
            if self.first_n_grad_im > 0:
                grad_diff_im = energy_scale*(self.gradient_sigma(sigma_predicted[:,:,self.nw:])-self.gradient_sigma(sigma_true[:,:,self.nw:]))
                loss += (self.gradient_weight)*torch.mean(grad_diff_im[:,:,:self.first_n_grad_im]**2)

            # if self.debug:
            #     print_loss_im = grad_diff_im.cpu().detach().numpy()
            #     print_loss_re = grad_diff_re.cpu().detach().numpy()
            #     print('FD gradient penalty (re, im): ', print_loss_re, print_loss_im)

        return loss
