"""
========================
Instructions
========================

Use the ``mlgf.data.Dataset`` class for loading and manipulating data.

Example::

    from mlgf.data import Dataset
    import numpy as np
    
    dyn_imag_freq_points = np.array([1e-3, 0.1, 0.2, 0.5, 1.0, 2.0])
    DOS_drop_outs = ['mol0.joblib']
    
    dset = Dataset.from_srcdirectory(src,
        dropout_files = DOS_drop_outs,
        data_format='joblib'
        )
        
    dset = dset.choose_random_subset(
        ml_samples, random_seed=seed
        )
    
    dset = dset.prep_dyn_features(dyn_imag_freq_points)
    
    trainset, testset = dset.get_train_test_split(
        test_frac=0.3, 
        random_seed=seed
        )
    
    feat_list = ['P', 'Fock', 'Hcore', 'Vj', 'Vk', 'gf_dyn']

    x_train = trainset.get_offdiag_features(feat_list)
    y_train = trainset.get_offdiag_features(['sigma_saiao'])
"""

from ._dataset import Dataset
from ._data import Data
__all__ = ['Dataset', 'Data']
