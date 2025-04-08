import pickle
import joblib
import copyreg
from functools import cache
import pyscf.dft, pyscf.lib
from pyscf.gto import Mole
import numpy as np

from mlgf.utils.linalg import BasisChanger, mat_type_map
from mlgf.lo.saao import get_saao
from mlgf.lib.ml_helper import get_custom_freq_gfhf_features, get_hyb_off, gGW_mo_saiao, get_sigma_fit, exclude_core_ftrs, get_g0
from mlgf.lib.ml_helper import get_core_orbital_indices, get_orbtypes_df, get_saiao_charges, get_saiao_locality
from mlgf.lib.ml_helper import get_saiao_features, get_saao_features, get_chk_saiao

def load_md_saiao(mlf, fname, force_eigv_direction = True, val_core_dats = None):

    mol = pyscf.lib.chkfile.load_mol(fname)
    
    if val_core_dats is not None:
        ele_symbols = list(set([a[0] for a in mol._atom]))
        new_core_data = {}
        for ele_key in val_core_dats[1].keys():
            if ele_key in ele_symbols:
                new_core_data[ele_key] = val_core_dats[1][ele_key]

        new_val_core_dats = [val_core_dats[0], new_core_data]

    else:
        new_val_core_dats = val_core_dats
    
    
    mf = pyscf.dft.RKS(mol)
    mf.xc = mlf.get('xc', 'hf')
    if type(mf.xc) is bytes:
        mf.xc = mf.xc.decode('utf-8')
    scf_data = pyscf.lib.chkfile.load(fname, 'scf')
    mf.__dict__.update(scf_data)
    C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao(mf, mlf['fock'], minao = 'minao', force_eigv_direction = force_eigv_direction, val_core_dats = new_val_core_dats)
    C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)
    mlf = get_saiao_features(mol, mlf, C_ao_saiao)

    mlf['fock_iao'] = fock_iao
    mlf['C_ao_iao'] = C_ao_iao
    mlf['C_iao_saiao'] = C_iao_saiao
    mlf['hcore+vj_saiao'] = mlf['hcore_saiao'] + mlf['vj_saiao']
    mlf['inds_core'] = get_core_orbital_indices(mol)

    df_mol = get_orbtypes_df(mol)
    mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])
    mlf['atomic_charge_saiao'] = get_saiao_charges(df_mol, mlf['dm_saiao'])
    mlf['boys_saiao'] = get_saiao_locality(mol, mlf['C_ao_saiao'])
    return mlf

def load_md_saao(mlf, fname, force_eigv_direction = True):

    mol = pyscf.lib.chkfile.load_mol(fname)
    mlf['inds_core'] = get_core_orbital_indices(mol)
    C_ao_saao = get_saao(mol, mlf['dm_hf'], force_eigv_direction = force_eigv_direction)
    mlf = get_saao_features(mol, mlf, C_ao_saao)
    mlf['C_ao_saao'] = C_ao_saao
    mlf['hcore+vj_saao'] = mlf['hcore_saao'] + mlf['vj_saao']
    
    df_mol = get_orbtypes_df(mol)
    mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])
    mlf['atomic_charge_saao'] = get_saiao_charges(df_mol, mlf['dm_saao'])
    mlf['boys_saao'] = get_saiao_locality(mol, mlf['C_ao_saao'])
    return mlf

@cache
def get_triu_indices(n):
    # Cached to avoid repeated calls to np.triu_indices
    return np.triu_indices(n, k=1, m=None)

def reduce_mole(mol):
    # pyscf Mole objects are not pickleable
    # probably should submit a patch to pyscf
    # See https://docs.python.org/3/library/pickle.html#pickling-class-instances
    return Mole.loads, (Mole.dumps(mol),)

def array_summary(obj):
    if isinstance(obj, np.ndarray):
        return f'array(shape={obj.shape}, dtype={obj.dtype})'
    else:
        return str(obj)

def realify(mat):
    if mat.dtype == np.complex128:
        return np.column_stack([mat.real, mat.imag])
    else:
        return mat

class DataPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[Mole] = reduce_mole

class Data:
    # Stores data from SCF calculation on a single molecule
    # This is a wrapper around a dict.
    # You can access its fields with dot notation like Pandas.
    _internal_names_set = {'data'}
    def __setattr__(self, name, value):
        if name in self._internal_names_set:
            super().__setattr__(name, value)
        else:
            self.data[name] = value
    
    def __getattr__(self, name):
        if name not in self._internal_names_set:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f'Attribute {name} not found')
        return super().__getattribute__(name)
    
    def __getitem__(self, key):
        if key == 'mol':
            if not self.data['mol']._built:
                self.data['mol'].build(verbose=0)
        return self.data[key]
    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)
    def __delitem__(self, key):
        self.data.__delitem__(key)
    def __contains__(self, key):
        return (key in self.data)
    
    def keys(self):
        return self.data.keys()
    
    
    def __init__(self, idict):
        self.data = dict(idict)
    
    @staticmethod
    def from_dict(idict):
        return Data(idict)
    
    def __repr__(self):
        return "Data(\n" + ",\n".join(str(k) + "=" + array_summary(v) for k, v in self.data.items()) + ")"

    def __str__(self):
        return self.__repr__()


    @staticmethod
    def from_kwargs(**kwargs):
        # Initialize from keyword arguments
        return Data(kwargs)


    def get_mf(self, scf_constructor=pyscf.dft.RKS):
        """Creates a pyscf mean-field object from the saved data

        Args:
            scf_constructor (function, optional): Constructor for mean-field object. Defaults to pyscf.dft.RKS.

        Returns:
            mf: mean-field object
        """
        if('mol' in self.data):
            mf = scf_constructor(self.mol, xc='hf')
            if 'e_tot' in self.data:
                mf.__dict__.update({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                                    'mo_occ': self.mo_occ, 'mo_coeff': self.mo_coeff})
            else:
                mf.__dict__.update({'mo_energy': self.mo_energy, 'mo_occ': self.mo_occ,
                                    'mo_coeff': self.mo_coeff})
                
            return mf
        else:
            raise ValueError('You need to set the mol attribute before calling get_mf')
    
    def basis_setup(self, bas_list):
        # Set up basis
        if 'bc_ao_mo' not in self.data:
            self.bc_ao_mo = BasisChanger(self.data['ovlp'], self.data['mo_coeff'])
        
        for bas in bas_list:
            if f'bc_ao_{bas}' not in self.data:
                self.data[f'bc_ao_{bas}'] = BasisChanger(self.data['ovlp'], self.data[f'C_ao_{bas}'])
                self.data[f'S_{bas}'] = self.data[f'bc_ao_{bas}'].rotate_focklike(self.data['ovlp'])
                self.data[f'bc_mo_{bas}'] = self.bc_ao_mo.inverse().chain(self.data[f'bc_ao{bas}'])
    
    def ftr_as(self, ftr, bas_orig, bas_new, mat_type=None):
        # try to see ftr in basis bas
        newbases = {bas_orig, bas_new}.difference({'ao', 'mo'})
        self.basis_setup(newbases)
        rev=False
        if f'bc_{bas_orig}_{bas_new}' in self.data:
            bc = self.data[f'bc_{bas_orig}_{bas_new}']
        elif f'bc_{bas_new}_{bas_orig}' in self.data:
            bc = self.data[f'bc_{bas_new}_{bas_orig}']
            rev=True
        else:
            bc = self.data[f'bc_ao_{bas_orig}'].inverse().chain(self.data[f'bc_ao_{bas_new}'])
            self.data[f'bc_{bas_orig}_{bas_new}'] = bc
        
        if not mat_type:
            mat_type = mat_type_map[ftr]
        
        return bc.transform(self.data[ftr], mat_type=mat_type, rev=rev)
        
        
    def save(self, f):
        if hasattr(f, 'write'):
            DataPickler(f, protocol=5).dump(self.data)
        else:
            with open(f, 'wb') as fp:
                DataPickler(fp, protocol=5).dump(self.data)
    
    def save_joblib(self, f):
        md2 = self.data.copy()
        if 'mol' in md2:
            md2['mol'] = Mole.dumps(md2['mol'])
        joblib.dump(md2, f, protocol=5)
    
    @staticmethod
    def load_joblib(f):
        md = Data(joblib.load(f))
        if 'mol' in md.data:
            if isinstance(md.data['mol'], str):
                    md.data['mol'] = Mole.loads(md.data['mol'])
        md.fname = f
        return md
    
    @staticmethod
    def load_chk(f, purge_keys = [], force_eigv_direction = True, val_core_dats = None, symmetrize_sigmaI = True, basis = 'saiao'):
        """primary load function for Data objects (from chk files)

        Args:
            f (str): .chk file name 
            purge_keys (list, optional): attributes to remove if wanting to save memory. Defaults to [].
            force_eigv_direction (bool, optional): force the SAIAO rotation to force the eigenvector directions. Defaults to True.
            val_core_dats (list, optional): data for redefining core orbitals for seperate projections. Defaults to None.
            symmetrize_sigmaI (bool, optional): symmetryize the sigma(iw) matrices (e.g. if not symmetry for linear solved sigma). Defaults to True.

        Returns:
            Data: the resulting Data for the molecule, with SAIAO features in numpy format
        """      
        assert(basis in ['saiao', 'saao'])  
        mlf = pyscf.lib.chkfile.load(f, 'mlf')
        if symmetrize_sigmaI and 'dm_cc' in mlf.keys():
            gf0 = get_g0(mlf['omega_fit'], mlf['mo_energy'], 0.)
            gf = np.linalg.inv(np.linalg.inv(gf0.T) - mlf['sigmaI'].T).T
            gf = (gf + np.transpose(gf, axes=(1, 0, 2)))/2
            mlf['sigmaI'] = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()
            del gf
            del gf0

        if symmetrize_sigmaI and 'sigmaI' in mlf.keys():
            mlf['sigmaI'] = (mlf['sigmaI'] + np.transpose(mlf['sigmaI'], axes=(1, 0, 2)))/2

        if basis == 'saiao':
            mlf = load_md_saiao(mlf, f, force_eigv_direction = force_eigv_direction, val_core_dats = val_core_dats)
        if basis == 'saao':
            mlf = load_md_saao(mlf, f, force_eigv_direction = force_eigv_direction)

        md = Data({key: value for key, value in mlf.items() if key not in purge_keys})
        if 'mol' in md.data:
            if isinstance(md.data['mol'], str):
                    md.data['mol'] = Mole.loads(md.data['mol'])
        md.fname = f
        md.basis = basis
        return md
    
    @staticmethod
    def load(f):
        # Load data from a file.
        # Takes a path-like or file-like object
        if hasattr(f, 'read'):
            return Data(pickle.load(f))
        else:
            # assume f is a path-like object
            with open(f, 'rb') as fp:
                return Data(pickle.load(fp))
    
    # Calculate dynamical features
    def calc_dyn(self, dyn_imag_freq_points, ftr_suffix = '', add_ef = True):

        if add_ef:
            dyn_imag_freq_points = self.ef.copy() + dyn_imag_freq_points.copy()
        gf_dyn, hyb_dyn = get_custom_freq_gfhf_features(
            self.mo_energy, self[f'fock_{self.basis}'], self[f'C_{self.basis}_mo'], dyn_imag_freq_points, mlf_chkfile = getattr(self, 'fname', ''))
        setattr(self, f'gf_dyn{ftr_suffix}', gf_dyn)
        setattr(self, f'hyb_dyn{ftr_suffix}', hyb_dyn)

        hyb_dyn_off = get_hyb_off(self[f'fock_{self.basis}'], gf_dyn, dyn_imag_freq_points)
        setattr(self, f'hyb_dyn_off{ftr_suffix}', hyb_dyn_off)
        # try:
        #     setattr(self, 'hcore+vj_saiao', self.hcore_saiao + self.vj_saiao)
        #     # setattr(self, 'hcore_inv_saiao', np.log10(np.abs(1./self.hcore_saiao)))
        #     # setattr(self, 'vj_inv_saiao', np.log10(np.abs(1./self.vj_saiao)))
        # except AttributeError:
        #     pass

    def refit_sigma(self, omega_fit):
        # Calculate dynamical features
        assert(len(self.freqs) >= len(omega_fit))
        setattr(self, 'omega_fit', omega_fit)
        sigma_fit = get_sigma_fit(self.sigmaI, self.freqs, omega_fit, freq_tol = 1.0e-8)
        for basis in ['saao', 'saiao']:
            basis_mo_rotation = f'C_mo_{basis}' 
            if basis_mo_rotation in self.data:
                sigma_fit_lmo = gGW_mo_saiao(sigma_fit, self.data[basis_mo_rotation])
                setattr(self, f'sigma_{basis}', sigma_fit_lmo)        

    def get_diag_features(self, features, ret_labels=False, exclude_core = False):
        #return np.column_stack([np.diagonal(getattr(self, f)).T for f in features])
        if exclude_core: 
            inds_core = self.inds_core
        to_stack = []
        for f in features:
            if 'hyb_dyn' in f:
                feat = self.data[f]
                if exclude_core: 
                    feat = exclude_core_ftrs(feat, inds_core, rank2 = False)
            else:
                feat = np.diagonal(self.data[f]).T
                if exclude_core: 
                    feat = exclude_core_ftrs(feat, inds_core, rank2 = False)
            to_stack.append(realify(feat))
        diag_features = np.column_stack(to_stack)
        
        if ret_labels:
            labels = []
            for f in features:
                feat = self.data[f]
                if feat.ndim == 3:
                    ncol = feat.shape[2]
                elif f == 'hyb_dyn':
                    ncol = feat.shape[1]
                else:
                    # simple diagonal
                    ncol = 1
                
                if feat.dtype == np.complex128:
                    for part in ('re', 'im'): # real part, then imaginary part
                        for i in range(ncol):
                            labels.append(f'{f}_{i}_{part}' if ncol > 1 else f'{f}_{part}')
                else: # pure real
                    for i in range(ncol):
                        labels.append(f'{f}_{i}' if ncol > 1 else f)
            return diag_features, labels
        else:
            return diag_features
                        
    
    
    def get_offdiag_features(self, features, ret_labels = False, coulomb_screen_tol = None, coulomb_screen_basis = 'saiao', exclude_core = False):
        nmo = self.mo_coeff.shape[0]
        if exclude_core:
            iu = get_triu_indices(nmo - len(self.inds_core))
        else:
            iu = get_triu_indices(nmo)
        
        def unravel_(mat):
            if mat.ndim == 3:
                return mat[iu[0], iu[1], :]
            else:
                return mat[iu]
        
        if coulomb_screen_tol is not None: 
            # print(f'Screening coulomb matrix np.abs(vj_{coulomb_screen_basis}) < {coulomb_screen_tol}')
            screen = np.abs(self.data[f'vj_{coulomb_screen_basis}']) > coulomb_screen_tol
           
        else:
            screen = np.ones((nmo, nmo), dtype = bool)

        if exclude_core:
            inds_core = self.inds_core
            # print(screen.shape, inds_core)
            screen = unravel_(exclude_core_ftrs(screen, inds_core))
            offdiag_features = np.column_stack([realify(unravel_(exclude_core_ftrs(self.data[f], inds_core))[screen]) for f in features])
        else:
            screen = unravel_(screen)
            offdiag_features = np.column_stack([realify(unravel_(self.data[f])[screen]) for f in features])
        
        if ret_labels:
            labels = []
            for f in features:
                feat = self.data[f]
                if feat.ndim == 3:
                    ncol = feat.shape[2]
                else:
                    ncol = 1
                
                if feat.dtype == np.complex128:
                    for part in ('re', 'im'): # real part, then imaginary part
                        for i in range(ncol):
                            labels.append(f'{f}_{i}_{part}' if ncol > 1 else f'{f}_{part}')
                else: # pure real
                    for i in range(ncol):
                        labels.append(f'{f}_{i}' if ncol > 1 else f)
            return offdiag_features, labels
        else:
            return offdiag_features
    
    # needs handling of dynamical quantities gf_dyn and hyb_dyn
    def calc_ii_jj_features(self, features):
        for f in features:
            mat = self.data[f]
            vec = np.diagonal(mat).T
            
            # A_ii + A_jj
            setattr(self, f'{f}_iijj_plus', np.add.outer(vec, vec))
            
            # |A_ii - A_jj|
            setattr(self, f'{f}_iijj_minus', np.add.outer(vec, vec))
            
        return self


