import os
import numpy as np
import pandas as pd
from ._data import Data
import pathlib
import re
import warnings


def get_conf_nums(fnames):
    stems = [pathlib.Path(f).stem for f in fnames]
    
    try:
    # This regex gets the last contiguous sequence of digits in a string.
    # The string must end in a digit.
        conf_nums = [int(re.match(r'.*?(\d+)\Z', stem).group(1)) for stem in stems]
    except AttributeError:
        # warnings.warn(f'Could not parse numbers from file stems {stems}')
        return [i for i in range(len(stems))]
    return conf_nums
    

class Dataset:
    """Holds a list of data from DFT and/or GW calculations as Data for each DFT or DFT+GW calculation
    Attributes:
        fnames (list): List of names of data files
        loaded (dict): Dictionary containing loaded data
        data_format (str): Format of data files
        
    """
    _format_extension_tbl = {'joblib': '.joblib', 'internal': '.mdt', 'chk' : '.chk'}
    
    
    def __init__(self, iterable, loaded = {}, data_format = 'chk', preserve_order=True, load_data = True, purge_keys = [], core_projection_file_path = None, basis = 'saiao'):
        if preserve_order:
            self.fnames = list(iterable)
        else:
            self.fnames = sort_fnames_by_number(iterable)
        self.conf_nums = list(get_conf_nums(self.fnames))
        self.conf_nums_to_fnames = dict(zip(self.conf_nums, self.fnames))
        self.purge_keys = purge_keys
        self.basis = basis
       
        self.confs_are_unique = (len(self.conf_nums_to_fnames) == len(self.conf_nums))
        
        if not self.confs_are_unique:
            warnings.warn('Conf numbers are not unique.')
        
        self.loaded = loaded
        self.data_format = data_format
        self.load_data = load_data
        if self.data_format not in Dataset._format_extension_tbl.keys():
            raise ValueError(f'Unknown data format {self.data_format}; acceptable values are {Dataset._format_extension_tbl.keys()}')
        if not core_projection_file_path is None:
            minao_val = core_projection_file_path + '/minao_val.dat'
            minao_core = {'Si': core_projection_file_path + '/minao_core.dat', 'C' : core_projection_file_path + '/minao_core.dat', 'O' : core_projection_file_path + '/minao_core.dat'}
            self.val_core_dats = [minao_val, minao_core]
        else:
            self.val_core_dats = None

    
    def get_by_fname(self, fname):
        if fname in self.loaded:
            return self.loaded[fname]
        else:
            if self.data_format == 'joblib':
                dat = Data.load_joblib(fname)
            elif self.data_format == 'chk':
                dat = Data.load_chk(fname, purge_keys = self.purge_keys, val_core_dats = self.val_core_dats, basis = self.basis)
            elif self.data_format == 'internal':
                dat = Data.load(fname)
            else:
                raise ValueError(f'Unknown data format {self.data_format}; acceptable values are {Dataset._format_extension_tbl.keys()}')
            if self.load_data:
                self.loaded[fname] = dat
            return dat
    
    def get_by_confnum(self, confnum):
        if not self.confs_are_unique:
            raise ValueError('Conf numbers are not unique.')
        else:
            return self.get_by_fname(self.conf_nums_to_fnames[confnum])
        
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        return self.get_by_fname(fname)

    
    def __setitem__(self, idx, value):
        self.loaded[self.fnames[idx]] = value
    
    def __len__(self):
        return len(self.fnames)
    
    def __iter__(self):
        return (self.get_by_fname(fname) for fname in self.fnames)
    
    @staticmethod
    def from_files(file_list, dropout_files = [], data_format = 'chk', load_data = True, purge_keys = [], core_projection_file_path = None, basis = 'saiao'):
        return Dataset([filename for filename in file_list if filename not in dropout_files], data_format=data_format, load_data = load_data, purge_keys = purge_keys, core_projection_file_path = core_projection_file_path, basis = basis)
            
    @staticmethod
    def from_srcdirectory(src, dropout_files = [], data_format = 'chk'):
        extension = Dataset._format_extension_tbl[data_format]
        filenames = [f for f in os.listdir(src) if f.endswith(extension) and f not in dropout_files]
        return Dataset([os.path.join(src, filename) for filename in filenames], data_format=data_format)
    
    
    def get_subset(self, indices, load_data = True):
        """Takes a subset of the dataset, specified by indices.

        Args:
            indices (list(int)): list of indices

        Returns:
            Dataset: subset specified by indices
        """
        indices = sorted(indices)
        fnames = [self.fnames[idx] for idx in indices]

        return Dataset(fnames,
                loaded = {fname: self.loaded[fname] for fname in fnames if fname in self.loaded},
                data_format=self.data_format, load_data = load_data
        )

    def get_subset_from_file_list(self, file_list, load_data = True):
        indices = [ind for ind in range(len(self)) if self[ind].fname in file_list]
        return self.get_subset(indices, load_data = load_data)

    def choose_random_subset(self, num_confs, random_seed = 0):
        """Returns a size num_confs subset of the dataset, chosen randomly.

        Args:
            num_confs (int): Size of desired subset
            random_seed (int, optional): Random seed. Defaults to 0.

        Returns:
            Dataset: random subset
        """
        assert(num_confs <= len(self.fnames))
        rng = np.random.default_rng(random_seed)
        subset_indices = np.sort(rng.choice(len(self.fnames), num_confs, replace = False))
        return self.get_subset(subset_indices)
    
    def get_train_test_split(self, test_frac = 0.3, random_seed = 0, return_indices = False):
        rng = np.random.default_rng(random_seed)
        
        train_cut = int((1-test_frac)*len(self.fnames))
        # test_cut = len(self.fnames) - train_cut
        
        # Random permutation of 1...N
        shuf_inds = list(rng.permutation(len(self.fnames)))
        
        
        training_idx, test_idx = shuf_inds[:train_cut], shuf_inds[train_cut:]
        
        dset_train, dset_test = self.get_subset(training_idx), self.get_subset(test_idx)
        
        if return_indices:
            return dset_train, dset_test, training_idx, test_idx
        else:
            return dset_train, dset_test
    
    def num_orbs(self):
        return self[0]['P'].shape[0]
    
    # prepare dynamical features on supplied frequency grid (real or imag), ftr_suffix will be added to the end of the feature names of the new dynamical features for retrieval purposes
    # add_ef shifts the frequency points by the real valued fermi level (tyipically true for an imaginary grid and false otherwise)
    def prep_dyn_features(self, dyn_imag_freq_points, ftr_suffix = '', add_ef = True):
        setattr(self, f'dyn_freq_points{ftr_suffix}', dyn_imag_freq_points)
        setattr(self, f'dyn_freq_points{ftr_suffix}_ef', add_ef)
        for i in range(len(self.fnames)):
            self[i].calc_dyn(dyn_imag_freq_points, ftr_suffix = ftr_suffix, add_ef = add_ef)
        return self
        
    def prep_ii_jj_features(self, features):
        for i in range(len(self.fnames)):
            self[i].calc_ii_jj_features(features)
        return self

    def reprep_sigma_fit(self, omega_fit):
        for i in range(len(self.fnames)):
            self[i].refit_sigma(omega_fit)
        return self
    
    def get_diag_features(self, features, as_dataframe=False, exclude_core = False):
        assert(len(features) > 0)
        assert(len(self.fnames) > 0)
        if as_dataframe:
            _, feature_labels = self[0].get_diag_features(features, ret_labels=True)
            features = np.row_stack([self[i].get_diag_features(features, exclude_core = exclude_core) for i in range(len(self.fnames))])
            return pd.DataFrame(features, columns=feature_labels)
        else:
            return np.row_stack([self[i].get_diag_features(features, exclude_core = exclude_core) for i in range(len(self.fnames))])
    
    def get_offdiag_features(self, features, as_dataframe = False, coulomb_screen_tol = None, coulomb_screen_basis = 'saiao', exclude_core = False):
        assert(len(features) > 0)
        assert(len(self.fnames) > 0)

        if as_dataframe:
            _, feature_labels = self[0].get_offdiag_features(features, ret_labels=True)
            features = np.row_stack([self[i].get_offdiag_features(features, coulomb_screen_tol = coulomb_screen_tol, coulomb_screen_basis = coulomb_screen_basis, exclude_core = exclude_core) for i in range(len(self.fnames))])
            return pd.DataFrame(features, columns=feature_labels)
        else:
            return np.row_stack([self[i].get_offdiag_features(features, coulomb_screen_tol = coulomb_screen_tol, coulomb_screen_basis = coulomb_screen_basis, exclude_core = exclude_core) for i in range(len(self.fnames))])

def sort_fnames_by_number(fnames):
    conf_nums = get_conf_nums(fnames)
    fnames_sorted = [fname for _, fname in sorted(zip(conf_nums, fnames))]
    return fnames_sorted

