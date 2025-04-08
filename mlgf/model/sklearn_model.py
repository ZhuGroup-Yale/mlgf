from mlgf.data import Dataset, Data
from mlgf.lib.ml_helper import get_sigma_from_ml

import os
import numpy as np
import argparse
import joblib
import time
import warnings
import psutil

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, PairwiseKernel, DotProduct, WhiteKernel, Matern, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn

import gc

# -1 uses max cores avail
def predict_joblib(model, x, n_jobs = -1, backend = 'threading', max_mem = 2000):
    print('Memory max for joblib parallel backend: ', max_mem, 'MB')
    with joblib.parallel_backend(backend = backend, n_jobs = n_jobs, max_nbytes = f'{max_mem}M'):
        y = model.predict(x)
    return y

def fit_joblib(model, x, y, n_jobs = -1, backend = 'threading', max_mem = 2000):
    print('Shape of x passed to fit_joblib: ', x.shape)
    print('Memory max for joblib parallel backend: ', max_mem, 'MB')
    with joblib.parallel_backend(backend = backend, n_jobs = n_jobs, max_nbytes = f'{max_mem}M'):
        return model.fit(x, y)

"""
KRROrchestrator is the central object for training KRR model and predicting with it
"""
class SKL_Model:

    def __init__(self, feature_list_ii, feature_list_ij, target, dyn_imag_freq_points = [], kernel_ii = None, kernel_ij = None, alpha_ii = None, alpha_ij = None,
        transformer_xii = None, transformer_yii = None, transformer_xij = None, transformer_yij = None, regressor_ii = 'krr', regressor_ij = 'krr',
        exclude_core = False, coulomb_screen_tol = None, coulomb_screen_basis = 'saiao', max_mem = 5000, ij_chunk_size = 1e3):

        default_kernel = 1.0 \
        + Matern(length_scale=1.0, length_scale_bounds='fixed') \
        + 0.5*Matern(length_scale=0.5, length_scale_bounds='fixed') \
        + 0.25*Matern(length_scale=0.1, length_scale_bounds='fixed') \

        default_alpha_ii, default_alpha_ij = 0.0000100, 0.0000100
        default_transformer_xii, default_transformer_yii = MinMaxScaler(), MinMaxScaler()
        default_transformer_xij, default_transformer_yij = MinMaxScaler(), MinMaxScaler()

        if regressor_ii not in ['krr', 'gpr']:
            raise NotImplementedError('Regressor_ii must be one of `krr` or `gpr`')

        if regressor_ij not in ['krr', 'gpr']:
            raise NotImplementedError('Regressor_ii must be one of `krr` or `gpr`')

        self.regressor_ii = regressor_ii
        self.regressor_ij = regressor_ij

        self.feature_list_ii = feature_list_ii
        self.feature_list_ij = feature_list_ij
        self.target = target
        self.feat_list_iijj = list(set(['_'.join(f.split('_')[:2]) for f in feature_list_ij if 'iijj' in f]))

        self.exclude_core = exclude_core # only relevant for training purposes, not prediction
        self.coulomb_screen_tol = coulomb_screen_tol
        self.coulomb_screen_basis = coulomb_screen_basis
        self.dyn_imag_freq_points = dyn_imag_freq_points

        if kernel_ii is None: 
            self.kernel_ii = sklearn.base.clone(default_kernel)
        else:
            self.kernel_ii = kernel_ii
        if kernel_ij is None:
            self.kernel_ij = sklearn.base.clone(default_kernel)
        else:
            self.kernel_ij = kernel_ij

        if alpha_ii is None: 
            self.alpha_ii = default_alpha_ii
        else:
            self.alpha_ii = alpha_ii
        if alpha_ij is None:
            self.alpha_ij = default_alpha_ij
        else:
            self.alpha_ij = alpha_ij

        if transformer_xii is None: 
            self.transformer_xii = default_transformer_xii
        else:
            self.transformer_xii = transformer_xii
        if transformer_yii is None:
            self.transformer_yii = default_transformer_yii
        else:
            self.transformer_yii = transformer_yii

        if transformer_xij is None: 
            self.transformer_xij = default_transformer_xij
        else:
            self.transformer_xij = transformer_xij
        if transformer_yij is None:
            self.transformer_yij = default_transformer_yij
        else:
            self.transformer_yij = transformer_yij 

        self.dyn_freq_points_list_ii = []
        self.dyn_freq_points_ef_list_ii = []
        self.dyn_freq_points_names_list_ii = []
        self.dyn_freq_points_list_ij = []
        self.dyn_freq_points_ef_list_ij = []
        self.dyn_freq_points_names_list_ij = []
        self.max_mem = max_mem
        self.ij_chunk_size = ij_chunk_size

    def fit_ii(self, dset, exclude_core = False):
        if self.regressor_ii == 'krr': self.model_ii = KernelRidge(kernel = self.kernel_ii, alpha = self.alpha_ii)
        if self.regressor_ii == 'gpr': self.model_ii = GaussianProcessRegressor(kernel = self.kernel_ii, alpha = self.alpha_ii)
        # if len(self.dyn_imag_freq_points) != 0 and np.sum(self.dyn_imag_freq_points - dset.dyn_imag_freq_points) > 1e-7:
        #     warnings.warn("Warning...........disagreement between existing model dyn_imag_freq_points and passed dset dyn_imag_freq_points")
        
        self.x_ii = dset.get_diag_features(self.feature_list_ii, exclude_core = self.exclude_core)
        self.y_ii = dset.get_diag_features([self.target], exclude_core = self.exclude_core)
        self.transformer_xii.fit(self.x_ii)
        self.transformer_yii.fit(self.y_ii)
        self.xt_ii = self.transformer_xii.transform(self.x_ii)
        self.yt_ii = self.transformer_yii.transform(self.y_ii)
        # self.model_ii.fit(self.xt_ii, self.yt_ii)
        self.model_ii = fit_joblib(self.model_ii, self.xt_ii, self.yt_ii, max_mem = self.max_mem)

    def fit_ij(self, dset):
        if self.regressor_ij == 'krr': self.model_ij = KernelRidge(kernel = self.kernel_ij, alpha = self.alpha_ij)
        if self.regressor_ij == 'gpr': self.model_ij = GaussianProcessRegressor(kernel = self.kernel_ij, alpha = self.alpha_ij)
        # if len(self.dyn_imag_freq_points) != 0 and np.sum(self.dyn_imag_freq_points - dset.dyn_imag_freq_points) > 1e-7:
        #     warnings.warn("Warning...........disagreement between existing model dyn_imag_freq_points and passed dset dyn_imag_freq_points")
        
        dset_dict = vars(dset)
        self.dyn_freq_points_list_ij = []
        self.dyn_freq_points_ef_list_ij = []
        self.dyn_freq_points_names_list_ij = []
        suffixes = list(set([f.split('_')[-1] for f in self.feature_list_ij if 'dyn' in f]))
        for key, val in dset_dict.items():
            if 'dyn_freq_points' in key and type(val) == np.ndarray and key.split('_')[-1] in suffixes:
                self.dyn_freq_points_list_ij.append(val)
                self.dyn_freq_points_names_list_ij.append(key)
                self.dyn_freq_points_ef_list_ij.append(dset_dict.get(f'{key}_ef', True))

        self.x_ij = dset.get_offdiag_features(self.feature_list_ij, coulomb_screen_tol = self.coulomb_screen_tol, coulomb_screen_basis = self.coulomb_screen_basis, exclude_core = self.exclude_core)
        self.y_ij = dset.get_offdiag_features([self.target], coulomb_screen_tol = self.coulomb_screen_tol, coulomb_screen_basis = self.coulomb_screen_basis, exclude_core = self.exclude_core)
        self.transformer_xij.fit(self.x_ij)
        self.transformer_yij.fit(self.y_ij)
        self.xt_ij = self.transformer_xij.transform(self.x_ij)
        self.yt_ij = self.transformer_yij.transform(self.y_ij)
        # self.model_ij.fit(self.xt_ij, self.yt_ij)
        self.model_ij = fit_joblib(self.model_ij, self.xt_ij, self.yt_ij, max_mem = self.max_mem)

    def fit(self, dset):
        t = time.time()
        print(f'Fitting ii elements...')
        self.fit_ii(dset)
        tii = time.time() - t
        print(f'Done fitting ii elements in {tii:0.2f}s, now fitting ij elements...')

        gc.collect()
        t = time.time()
        self.fit_ij(dset)
        tij = time.time() - t
        print(f'Done fitting! (ij fit in {tij:0.2f}s), Nii = {self.x_ii.shape[0]}, Nij = {self.x_ij.shape[0]}')
        self.tii, self.tij = tii, tij
        self.train_files = dset.fnames
        gc.collect()
        
    def predict_ii(self, dset, exclude_core, return_cov = False):
        xii = dset.get_diag_features(self.feature_list_ii, exclude_core = exclude_core)
        xii = self.transformer_xii.transform(xii)
        yii = predict_joblib(self.model_ii, xii, max_mem = self.max_mem)
        yii = self.transformer_yii.inverse_transform(yii)
        return yii

    def predict_ij(self, dset, exclude_core, coulomb_screen_tol = None, return_cov = False):
        xij_whole = dset.get_offdiag_features(self.feature_list_ij, exclude_core = exclude_core, coulomb_screen_tol = coulomb_screen_tol, coulomb_screen_basis = self.coulomb_screen_basis)
        nchunks = xij_whole.shape[0] // self.ij_chunk_size
        n_jobs = getattr(self, 'n_jobs', -1)
        if nchunks == 0: nchunks = 1
        chunks = np.array_split(xij_whole, nchunks)
        start_ind = 0
        print(f'nchunks = {nchunks}, n_jobs : {n_jobs}')
        for i, xij in enumerate(chunks):
            t0 = time.time()
            print(f'Chunking ij elements...chunk #: {i}')
            yij = predict_joblib(self.model_ij, self.transformer_xij.transform(xij), max_mem = self.max_mem, n_jobs = n_jobs)
            yij = self.transformer_yij.inverse_transform(yij)
            if start_ind == 0: yij_whole = np.empty((xij_whole.shape[0], yij.shape[1]))
            yij_whole[start_ind:start_ind+yij.shape[0], :] = yij
            start_ind = start_ind + yij.shape[0]
            t = time.time() - t0
            print(f'Chunk {i} took {t:02f}s')
        # yij_whole = self.transformer_yij.inverse_transform(self.model_ij.predict(self.transformer_xij.transform(xij_whole)))
        return yij_whole

    def mount_predict_dset(self, mlf_chkfile): # 
        file_list = [mlf_chkfile]
        self.pdset = Dataset.from_files(file_list, data_format='chk')
        self.pdset.prep_dyn_features(self.dyn_imag_freq_points)

    def predict_full_sigma(self, mlf_chkfile, remount_mlf_chkfile = True): # 
        if remount_mlf_chkfile:
            self.mount_predict_dset(mlf_chkfile) #
                            
        sigma_ii = self.predict_ii(self.pdset, self.exclude_core)
        sigma_ij = self.predict_ij(self.pdset, self.exclude_core)

        return get_sigma_from_ml(sigma_ii, sigma_ij)

    def dump(self, filename):
        joblib.dump(self, filename)