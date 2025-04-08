from functools import reduce
import numpy as np
import joblib
import warnings
import pandas as pd
import scipy

from pyscf import lib
from pyscf.lib import logger

from mlgf.lo.saao import get_C_ao_iao, get_saao

from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.ac.pade import _get_ac_idx

def get_mo_features(mlf):
    """get ML features in MO basis, not really used.

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py

    Returns:
        dict: modified mlf dictionary with MO basis features
    """    

    basis_name = 'mo'
    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    nocc = mlf['nocc']
    mo_occ = mlf['mo_occ']
            
    # feature 1: fock matrix
    fock_mo = np.diag(mo_energy)

    # feature 2 : density matrix
    dm_mo = np.diag(mo_occ)

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_mo = np.linalg.multi_dot((mo_coeff.T, hcore, mo_coeff))

    # features 4 & 5 : J and K matrices
    vj, vk = mlf['vj'], mlf['vk']
    vj_mo = np.linalg.multi_dot((mo_coeff.T, vj, mo_coeff))
    vk_mo = np.linalg.multi_dot((mo_coeff.T, vk, mo_coeff))

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    
    selected_freqs = mlf['omega_fit'] #ALREADY IMAG
    full_sigma = mlf['sigmaI'] # full sigma (>> len(omegaI))
    full_freqs = mlf['freqs']
    
    # ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2
    # omega = ef + selected_freqs 
    # g0_mo = get_g0(omega, mo_energy, eta=0)
    
    sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)

    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_mo
    mlf[f'fock_{basis_name}'] = fock_mo
    mlf[f'hcore_{basis_name}'] = hcore_mo
    mlf[f'vj_{basis_name}'] = vj_mo
    mlf[f'vk_{basis_name}'] = vk_mo
    mlf[f'sigma_{basis_name}'] = sigma_fit

    if 'vxc' in mlf.keys():
        vxc_mo = np.linalg.multi_dot((mo_coeff.T, mlf['vxc'], mo_coeff))
        mlf[f'vxc_{basis_name}'] = vxc_mo
    
    return mlf

def get_saao_features(mol, mlf, C_ao_saao):
    """get ML features in SAAO basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_saao (np.float64, norb x norb): rotation matrix from AO to SAAO

    Returns:
        dict: modified mlf dictionary with SAAO basis features
    """    


    basis_name = 'saao'
    mlf['C_ao_saao'] = C_ao_saao

    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nocc = mlf['nocc']
    nmo = len(mo_energy) #nao =  mol.nao_nr() #len(mo_energy)#
    dm = mlf['dm_hf']
    nelectron = nocc*2
    S_saao = np.linalg.multi_dot((C_ao_saao.T, S_ao, C_ao_saao))
            
    # feature 1: fock matrix
    fock = mlf['fock']
    fock_saao = np.linalg.multi_dot((C_ao_saao.T, fock, C_ao_saao))
    mo_energy_new, mo_coeff_new = scipy.linalg.eigh(fock_saao, S_saao)
    print('max mo energy diff: ', np.max(np.abs(mo_energy_new - mo_energy)))
    # print(mo_energy_new - mo_energy)
    assert ((np.max(np.abs(mo_energy_new - mo_energy)))< 1e-4)
    # transform Fock from SAAO back to AO
    S_saao_inv = np.linalg.inv(S_saao)
    SCS = np.linalg.multi_dot((S_saao_inv, C_ao_saao.T, S_ao))
    fock_ao_check = np.linalg.multi_dot((SCS.T, fock_saao, SCS))
    assert ((np.max(np.abs(fock_ao_check-fock)))<1e-6)

    # feature 2 : density matrix
    dm_saao = np.linalg.multi_dot((SCS, dm, SCS.T))
    assert(abs(np.trace(np.dot(dm_saao, S_saao))-mol.nelectron)<1e-8)

    # transform dm from SAAO back to AO
    dm_ao_new = np.linalg.multi_dot((C_ao_saao, dm_saao, C_ao_saao.T))
    assert ((np.max(np.abs(dm_ao_new-dm)))<1e-6)

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_saao = np.linalg.multi_dot((C_ao_saao.T, hcore, C_ao_saao))

    # features 4 & 5 : J and K matrices
    vj, vk = mlf['vj'], mlf['vk']
    vj_saao = np.linalg.multi_dot((C_ao_saao.T, vj, C_ao_saao))
    vk_saao = np.linalg.multi_dot((C_ao_saao.T, vk, C_ao_saao))

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2

    
    selected_freqs = mlf['omega_fit'] #ALREADY IMAG
    full_sigma = mlf['sigmaI'] # full sigma (>> len(omegaI))
    full_freqs = mlf['freqs']
    omega = ef + selected_freqs # CV note: omega set here
    g0_mo = get_g0(omega, mo_energy, eta=0)

    # MO to SAAO rotations for hamiltonian-like matrices
    C_saao_mo = np.linalg.multi_dot((S_saao_inv, C_ao_saao.T, S_ao, mo_coeff))
    C_mo_saao = np.linalg.multi_dot((mo_coeff.T, S_ao, C_ao_saao, S_saao_inv))
    
    # GF in SAAO basis
    g0_saao = np.zeros_like(g0_mo)
    
    sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)
    C_mo_saao = np.linalg.multi_dot((mo_coeff.T, S_ao, C_ao_saao))
    sigma_saao = gGW_mo_saiao(sigma_fit, C_mo_saao)
    
    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_saao
    mlf[f'fock_{basis_name}'] = fock_saao
    mlf[f'hcore_{basis_name}'] = hcore_saao
    mlf[f'vj_{basis_name}'] = vj_saao
    mlf[f'vk_{basis_name}'] = vk_saao
    mlf[f'gHF_{basis_name}'] = g0_saao
    mlf[f'sigma_{basis_name}'] = sigma_saao
    mlf[f'C_{basis_name}_mo'] = C_saao_mo
    mlf[f'C_mo_{basis_name}'] = C_saao_mo
    # mlf[f'C_mo_{basis_name}'] = C_mo_saao

    if 'vxc' in mlf.keys():
        vxc_saao = np.linalg.multi_dot((C_ao_saao.T, mlf['vxc'], C_ao_saao))
        mlf[f'vxc_{basis_name}'] = vxc_saao
    
    return mlf

def get_saiao_features(mol, mlf, C_ao_saiao, categorical = True):
    """get ML features in SAIAO basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_saao (np.float64, norb x norb): rotation matrix from AO to SAIAO
        categorical (boolean): whether to generate integer valued features for quantum numbers and orbital type

    Returns:
        dict: modified mlf dictionary with SAIAO basis features
    """    


    basis_name = 'saiao'
    mlf['C_ao_saiao'] = C_ao_saiao

    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nocc = mlf['nocc']
    nmo = len(mo_energy) #nao =  mol.nao_nr() #len(mo_energy)#
    dm = mlf['dm_hf']
    nelectron = nocc*2
            
    # feature 1: density matrix
    dm_saiao = np.linalg.multi_dot((C_ao_saiao.T, S_ao, dm, S_ao, C_ao_saiao))
    abs_diff_particle_number = abs(np.trace(dm_saiao)-nelectron)
    if abs_diff_particle_number > 1e-8:
        warnings.warn(f'dm_saiao particle number diff {abs_diff_particle_number:0.6e}')

    # feature 2 : Fock matrix
    fock = mlf['fock']
    fock_saiao = np.linalg.multi_dot((C_ao_saiao.T, fock, C_ao_saiao))

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_saiao = np.linalg.multi_dot((C_ao_saiao.T, hcore, C_ao_saiao))

    # feature 4 & 5 : J and K matrices
    vj, vk = mlf['vj'], mlf['vk']
    vj_saiao = np.linalg.multi_dot((C_ao_saiao.T, vj, C_ao_saiao))
    vk_saiao = np.linalg.multi_dot((C_ao_saiao.T, vk, C_ao_saiao))

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2

    # GF in SAIAO basis
    C_saiao_mo = np.linalg.multi_dot((C_ao_saiao.T, S_ao, mo_coeff))
    C_mo_saiao = C_saiao_mo.T
    # With ndim>2 arrays, numpy matmul uses the last two axes

    
    try:
        selected_freqs = mlf['omega_fit'] #ALREADY IMAG
        full_sigma = mlf['sigmaI'] # full sigma (>> len(omegaI))
        full_freqs = mlf['freqs']
        omega = ef + selected_freqs # CV note: omega set here

        sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)
        sigma_saiao = gGW_mo_saiao(sigma_fit, C_mo_saiao)
        mlf[f'sigma_{basis_name}'] = sigma_saiao

    except KeyError as e:
        # print('Not generating saiao features for sigma and G: ', str(e))
        pass
    
    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_saiao
    mlf[f'fock_{basis_name}'] = fock_saiao
    mlf[f'hcore_{basis_name}'] = hcore_saiao
    mlf[f'vj_{basis_name}'] = vj_saiao
    mlf[f'vk_{basis_name}'] = vk_saiao
    mlf[f'C_{basis_name}_mo'] = C_saiao_mo

    if 'vxc' in mlf.keys():
        vxc_saiao = np.linalg.multi_dot((C_ao_saiao.T, mlf['vxc'], C_ao_saiao))
        mlf[f'vxc_{basis_name}'] = vxc_saiao

    if categorical:
        mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = get_orbtypes(mol)
        mlf['cat_orbtype_saiao'] = get_orb_type(mol, dm_saiao)
    return mlf

def get_sigma_ml(mlf_chkfile, pickle_file):
    """get ML sigma

    Args:
        mlf_chkfile (str): DFT calculation file
        pickle_file (str): joblib file with model object

    Returns:
        np.complex64: self-energy
    """    
    if  '.joblib' in pickle_file: 
        model_obj = joblib.load(pickle_file)
    sigma = model_obj.predict_full_sigma(mlf_chkfile)
    return sigma

def get_vmo(mlf):
    """get MO basis vk

    Args:
        mlf: DFT calculation dictionary

    Returns:
        np.float64, np.float64: vk_mo, vxc_mo
    """    
    mo_coeff, vk, vmf = mlf['mo_coeff'], mlf['vk_hf'], mlf['vxc']
    vk_mo = np.linalg.multi_dot((mo_coeff.T, vk, mo_coeff))
    vmf_mo = np.linalg.multi_dot((mo_coeff.T, vmf, mo_coeff))
    return vk_mo, vmf_mo

def get_pade18(nw=100):
    """get AC frequencies and weights from fcmdft

    Args:
        nw (int, optional): nomega for the initial GL grid. Defaults to 100.

    Returns:
        np.float64, np.float64: freqs, wts
    """    
    # get ef and freqs (to be consistent with GW)
    freqs, wts = _get_scaled_legendre_roots(nw=nw)
    freqs = np.concatenate([[0], freqs])
    nw_sigma = sum(freqs < 5.0) + 1
    freqs = freqs[:nw_sigma]
    idx = _get_ac_idx(len(freqs), idx_start=1)
    freqs = freqs[idx]
    wts = wts[idx]

    return freqs, wts

def get_triu_indices(n):
    # Cached to avoid repeated calls to np.triu_indices
    return np.triu_indices(n, k=1, m=None)

def unravel_rank3(mat):
    iu = get_triu_indices(mat.shape[0])
    if mat.ndim == 3:
        return mat[iu[0], iu[1], :]
    else:
        return mat[iu]

def find_subset_indices_numpy(fullset, subset, **kwargs):
    """get indices corresponding to subset in fullset, mainly for extracting AC indices 

    Args:
        fullset (np.float64)
        subset (np.float64)

    Raises:
        ValueError: if subset entry not in full set
        ValueError: finds multiple matches of a subset entry in full set

    Returns:
        np.array: indices
    """    
    assert(fullset.ndim == 1 and subset.ndim == 1)
    inds=[]
    for value in subset:
        match = np.flatnonzero(np.isclose(fullset, value, **kwargs))
        if match.size == 0:
            raise ValueError(f'Could not find {value} in {fullset}')
        elif match.size > 1:
            raise ValueError(f'Found multiple matches for {value} in {fullset}')
        inds.append(match[0])
    return np.array(inds, dtype=int)

# return GW in SAIAO basis from MO basis
# C^-1 = C^T (unitary in orthoganol basis ie SAIAO)
def gGW_mo_saiao_slow(gGW, C_mo_lo):
    gGW_saiao = np.zeros_like(gGW)
    for i in range(np.shape(gGW)[2]):
        gGW_saiao[:,:,i] = np.dot(C_mo_lo.T, gGW[:,:,i]).dot(C_mo_lo)
    return gGW_saiao

def gGW_mo_saiao(gGW, C_mo_lo):
    """GF rotation MO to local basis

    Args:
        gGW (np.array.complex64): greens function in MO basis
        C_mo_lo (np.array.float64): mo to local orbital basis rotation

    Returns:
        np.complex64: rotated GF
    """    
    
    return np.einsum('ji,jkn,kl->iln', C_mo_lo, gGW, C_mo_lo, optimize = True)

# return sigma in MO basis from localized orbital basis (SAAO or SAIAO)
def sigma_lo_mo(sigma_saiao, C_lo_mo):
    """GF rotation local basis to MO basis

    Args:
        gGW (np.array.complex64): greens function in local basis
        C_mo_lo (np.array.float64): local basis to MO basis rotation

    Returns:
        np.complex64: rotated GF
    """    
    return np.einsum('ji,jkn,kl->iln', C_lo_mo, sigma_saiao, C_lo_mo, optimize = True)

def get_g0(omega, mo_energy, eta):
    """get G0 (MF level) in MO basis

    Args:
        omega (np.array float64): frequencies on which to compute G0
        mo_energy (np.array float64): KS orbital energies
        eta (float): band-broadening

    Returns:
        np.array complex64: G0
    """    

    nmo = len(mo_energy)
    nw = len(omega)
    gf0 = np.zeros((nmo,nmo,nw),dtype=np.complex128)
    for iw in range(nw):
        gf0[:,:,iw] = np.diag(1.0/(omega[iw]+1j*eta - mo_energy))

    return gf0


def get_custom_freq_gfhf_features(mo_energy, fock_saiao, C_saiao_mo, omega, mlf_chkfile = ''):
    """diagonal dynamical features G0 and hyb

    Args:
        mo_energy (np.array float64): KS orbital energies
        fock_saiao (np.array float64): fock matrix in SAIAO
        C_saiao_mo (np.array float64): saiao to mo rotation matrix
        omega (np.array complex64): omega on which to generate dyn features
        mlf_chkfile (str, optional): file str to print if error. Defaults to ''.

    Returns:
        _type_: _description_
    """    
    # GF in MO basis on (ef + iw_n)
    g0_mo = get_g0(omega, mo_energy, eta=0)
    
    # GF in SAIAO basis   
    g0_saiao = np.einsum('ij,jkn,lk->iln', C_saiao_mo, g0_mo, C_saiao_mo, optimize = True)

    assert_traces = np.linalg.norm(np.trace(g0_mo[:,:,0])-np.trace(g0_saiao[:,:,0]))
    if assert_traces > 1e-8:
        warnings.warn(f'Warning! Trace of g0_mo and g0_saiao should be same (trace invariant under cyclic permutations), trace diff on w0: {assert_traces}, mlf_chkfile: {mlf_chkfile}')

    fock_diagonal = np.diagonal(fock_saiao)
    g0_diagonal = np.diagonal(g0_saiao)
    hyb_saiao = omega[:, None] - fock_diagonal - 1. / g0_diagonal
    hyb_saiao = hyb_saiao.T
    return g0_saiao, hyb_saiao

# fast pinv
# def stack_pinverse(mat_rank3):
#     """Compute the inverse of matrices in an array of shape (N,N,M)"""
#     return np.linalg.pinv(mat_rank3.transpose(2,0,1)).transpose(1,2,0)

def fast_pinv_2d(mats, det_tol = 1e-20):
    """quickly computes pinv of many 2x2 matrices using determinant, uses np.linalg.pinv (i.e. SVD Moorse-Penrose inverse) when det is near 0.
    speedup over all np.linalg.pinv should be ~8-fold

    Args:
        mats (np.float64): shape N x 2 x 2, the stack of 2 x 2 matrices
        det_tol (float, optional): near-zero tolerance for defaulting to np.linalg.pinv. Defaults to 1e-20.

    Returns:
        Nx2x2 stack of the pseudo inverses
    """    
    # assumes the matrices are symmetric
    try:
        import numexpr as ne
        a, b, c = mats[:,0,0].copy(), mats[:,1,1].copy(), mats[:,0,1].copy()
        dets = ne.evaluate('a*b-c**2')
        mask = np.abs(dets) > det_tol
        mats[mask,0,0] = b
        mats[mask,1,1] = a
        mats[mask,0,1] = - c
        mats[mask,1,0] = - c
        mat_masked = mats[mask,:,:]
        det_masked = dets[mask][:, np.newaxis, np.newaxis]
        mats[mask,:,:] = ne.evaluate('mat_masked / det_masked')
        mats[~mask] = np.linalg.pinv(mats[~mask])
    except ImportError:
        # numpy when numexpr not available
        dets = mats[:,0,0]*mats[:,1,1] - mats[:,0,1]*mats[:,1,0]
        mask = np.abs(dets) > det_tol
        a = mats[mask,0,0].copy()
        mats[mask,0,0] = mats[mask,1,1].copy()
        mats[mask,1,1] = a
        mats[mask,0,1] = - mats[mask,0,1]
        mats[mask,1,0] = - mats[mask,1,0]
        mats[mask,:,:] = mats[mask,:,:] / dets[mask][:, np.newaxis, np.newaxis]
        mats[~mask] = np.linalg.pinv(mats[~mask])
    return mats

def get_hyb_off(fock_saiao, g0_saiao, omega, mask = None):
    """Get hyb[ij] dynamical features

    Args:
        fock_saiao (np.array float64): fock in SAIAO
        g0_saiao (np.array complex64): g0 in saiao
        omega (np.array complex64): omega points on which to calculate the feature
        mask (np.array float64, optional): nmo x nmo mask to avoid computing full hyb_off (i.e. if edges are screened out), defaults to ones, i.e. fully compute hyb_off
    """    
    

    norbs = g0_saiao.shape[0]
    iu, il = np.triu_indices(norbs, 1), np.tril_indices(norbs, -1)
    n_ij = len(iu[0])
    if mask is None:
        mask = np.ones(n_ij, dtype=bool)
    else:
        mask = mask[iu[0], iu[1]]
    nomega = len(omega)
    offdiag = g0_saiao[iu]
    diag1 = g0_saiao[iu[0], iu[0], :]
    diag2 = g0_saiao[iu[1], iu[1], :]
    hyb_off_saiao = np.zeros((n_ij, nomega), dtype=np.complex_)
    stacks = np.zeros((n_ij, nomega, 2, 2), dtype=np.complex_)

    omega_minus_fock =  - np.tile(fock_saiao[iu], (len(omega),1)).T 
    
    stacks[:, :, 0, 0] = diag1
    stacks[:, :, 1, 1] = diag2
    stacks[:, :, 0, 1] =  offdiag
    stacks[:, :, 1, 0] =  offdiag

    for w in range(nomega):
        g0_inv = fast_pinv_2d(stacks[mask,w,:,:])[:,0,1]
        hyb_off_saiao[mask, w] = omega_minus_fock[mask,w] - g0_inv
    
    result = np.zeros(g0_saiao.shape, dtype = np.complex128)
    result[iu] = hyb_off_saiao
    result[il] = result.transpose((1, 0, 2))[il]

    return result

def get_sigma_fit(full_sigma, full_freqs, selected_freqs, freq_tol = 1.0e-8):
    """subset by frequencies

    Args:
        full_sigma (_type_): _description_
        full_freqs (_type_): _description_
        selected_freqs (_type_): _description_
        freq_tol (_type_, optional): _description_. Defaults to 1.0e-8.

    Returns:
        complex64: iomega subset of sigma
    """    
    assert(len(full_freqs)==full_sigma.shape[-1])
    sigma_subset = full_sigma[:,:,find_subset_indices_numpy(full_freqs, np.imag(selected_freqs), atol=freq_tol)]
    return sigma_subset.copy()

def get_chk_saiao(mf, fock_ao, minao = 'minao', force_eigv_direction = False, val_core_dats = None):
    """wrapper for localized IAO transform and symmetrization rotations 

    Args:
        mf (pyscf.mf): mean field object like DFT
        fock_ao (float64): ao basis fock
        minao (str, optional): which definition of MINAO to use. Defaults to 'minao'.
        force_eigv_direction (bool, optional): force the eigenvalue direction. Defaults to False.
        val_core_dats (list, optional): data for redefining MINAO native to pyscf. Defaults to None.

    Returns:
        tuple: C_ao_iao, C_iao_saiao, fock_iao
    """    
    if val_core_dats is None:
        C_ao_iao = get_C_ao_iao(mf, minao=minao)
    else:
        C_ao_iao = get_C_ao_iao(mf, minao=minao, minao_val=val_core_dats[0], minao_core=val_core_dats[1])
    fock_iao = reduce(np.dot, (C_ao_iao.T, fock_ao, C_ao_iao))
    C_iao_saiao = get_saao(mf.mol, fock_iao, force_eigv_direction = force_eigv_direction)
    
    assert(np.linalg.norm(np.dot(C_iao_saiao.T, C_iao_saiao)-np.eye(fock_iao.shape[0]))<1e-8)
    return C_ao_iao, C_iao_saiao, fock_iao

def get_chk_saao(mol, fock):  
    """only has symmetrization, no IAO

    Args:
        mol (pyscf.mol):
        fock (float64): AO fock matrix

    Returns:
        float64: AO to SAAO rotation
    """      
    C_ao_saao = get_saao(mol, fock)
    return C_ao_saao

def get_core_orbital_indices(mol):
    """get indicies of core orbs

    Args:
        mol (pyscf.mol):

    Returns:
        list: core index
    """    
    from mlgf.misc.config import config as core_config
    inds_core = []
    for core in core_config:
        inds_core = inds_core + list(mol.search_ao_label(core))
    return inds_core

def exclude_core_ftrs(feat, inds_core, rank2 = True):
    """removes core orb features

    Args:
        feat (float64): feature array
        inds_core (list): _description_
        rank2 (bool, optional): is the feature only rank2 (or rank3). Defaults to True.

    Returns:
        float64: feature array with core orbitals removed
    """    
    feat_new = np.delete(feat, inds_core, axis=0)
    if rank2: feat_new = np.delete(feat_new, inds_core, axis=1)
    return feat_new

def get_rsq(y_pred, y_true):
    """get rsq value between true and pred

    Args:
        y_pred (float64): pred
        y_true (float64): true

    Returns:
        float: rsq value
    """    
    ybar = np.mean(y_true)
    SST = np.sum((y_true - ybar)**2)
    SSReg = np.sum((y_pred - y_true)**2)
    return 1-SSReg/SST

def get_sigma_from_ml(sigma_ii, sigma_ij):
    """reshaping function for KRR

    Args:
        sigma_ii (float64): diag elements
        sigma_ij (float64): offdiag elements

    Returns:
        complex64: norb x norb x nomega
    """    
    nomega = sigma_ii.shape[-1]//2 
    norbs = sigma_ii.shape[0]
    sigma_ml_saiao = np.zeros((norbs, norbs, nomega), dtype=np.complex_)
    iu, il = np.triu_indices(norbs, 1), np.tril_indices(norbs, -1)
    for iw in range(nomega):     
        sigma_ml_saiao_iw = np.diag(sigma_ii[:,iw] + 1j*sigma_ii[:,nomega+iw])
        sigma_ml_saiao_iw[iu] = sigma_ij[:,iw] + 1j*sigma_ij[:,nomega+iw]
        sigma_ml_saiao_iw[il] = sigma_ml_saiao_iw.T[il]
        sigma_ml_saiao[:, :, iw] = sigma_ml_saiao_iw

    return sigma_ml_saiao

def rotate_sigma_uncertainty(sigma_uncertainty, C_lo_mo):
    """rotate uncertainty tensor from SAIAO to MO

    Args:
        sigma_uncertainty (complex64): uncertainty tensor
        C_lo_mo (float64): SAIAO to MO rotation

    Returns:
        complex64: norb x norb x nomega
    """    
    C_lo_mo = C_lo_mo**2
    return np.sqrt(sigma_lo_mo(sigma_uncertainty.real**2, C_lo_mo))+1j*np.sqrt(sigma_lo_mo(sigma_uncertainty.imag**2, C_lo_mo))

# def set_2orb_inds(mat, inds1, ind2, val):
#     idx = np.array(list(itertools.product(inds1, ind2)))
#     mat[tuple(np.array(idx).T)] = val # core-core
#     idx = np.array(list(itertools.product(ind2, inds1)))
#     mat[tuple(np.array(idx).T)] = val # core-core
#     return mat
    
def get_orb_type(mol, dm, virtual_tol = 1e-10):
    types = np.empty(dm.shape, dtype = 'int')
    
    inds_core = get_core_orbital_indices(mol)
    
    pao_cond = np.abs(np.diagonal(dm)) < virtual_tol
    inds_pao = np.where(pao_cond)[0]
    inds_iao = np.where(~pao_cond)[0]
    inds_iao = list(set(list(inds_iao)) - set(list(inds_core)))
     
    # types = set_2orb_inds(types, inds_core, inds_core, 0) # core-core
    # types = set_2orb_inds(types, inds_core, inds_iao, 1) # core-iao
    # types = set_2orb_inds(types, inds_core, inds_pao, 2) # core-pao
    # types = set_2orb_inds(types, inds_iao, inds_iao, 3) # iao-iao
    # types = set_2orb_inds(types, inds_iao, inds_pao, 4) # iao-pao
    # types = set_2orb_inds(types, inds_pao, inds_pao, 5) # pao-pao
    
    types[inds_pao, inds_pao] = 2 # pao
    types[inds_iao, inds_iao] = 1 # iao
    types[inds_core, inds_core] = 0 # core
    return types

def get_orb_principal(mol, dm):
    """converts princ number in pyscf mol string to an integer

    FIXME
    """    
    types = np.empty(dm.shape, dtype = 'int')
    inds = []
    chars = [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
    for char in chars:
        inds.append(list(mol.search_ao_label(char)))
    
    return np.diag(inds)

def get_ang_from_letter(letter):
    """converts angular letter in pyscf mol string to an integer

    Args:
        letter (str): spdf, etc

    Returns:
        int: l, -1 if failure
    """    
    if letter == 's':
        return 0
    if letter == 'p':
        return 1
    if letter == 'd':
        return 2
    if letter == 'f':
        return 3
    return -1

def get_orbtypes(mol):
    """orbital quantum numbers n and l

    Args:
        mol (pyscf.mol)

    Returns:
        tuple: n, l (both norb x norb)
    """    
    types_princ, types_ang = np.zeros(mol.nao), np.zeros(mol.nao)
    types_princ[:], types_ang[:] = -1, -1
    labs = mol.ao_labels()
    labs = [l.split(' ') for l in labs]
    for i in range(mol.nao):
        lab = labs[i]
        types_princ[i] = int(lab[2][0])-1
        types_ang[i] = get_ang_from_letter(lab[2][1])
    return np.diag(types_princ).astype('int'), np.diag(types_ang).astype('int')

def get_orbtypes_df(mol):
    """get orbital metadata as pandas table

    Args:
        mol (pyscf.mol)

    Returns:
        pd.DataFrame: metadata
    """    
    df = pd.Series(mol.ao_labels()).str.split(pat = ' ', n = 2, expand = True)
    df.columns = ['atm_id', 'atm_symbol', 'orbital_name']
    df['atm_id'] = df['atm_id'].astype('int')
    df['principal'] = df['orbital_name'].str[0].astype('int')
    df['angular'] = -1
    df.loc[df['orbital_name'].str.contains('s'), 'angular'] = 0
    df.loc[df['orbital_name'].str.contains('p'), 'angular'] = 1
    df.loc[df['orbital_name'].str.contains('d'), 'angular'] = 2
    df.loc[df['orbital_name'].str.contains('f'), 'angular'] = 3
    ids = np.unique(df['atm_id'])
    charge_metadata = pd.DataFrame({'atm_id' : ids})
    charge_metadata['nuclear_charge'] = [mol.atom_charge(i) for i in list(ids)]
    df = df.merge(charge_metadata, on = 'atm_id')
    return df 

def get_saiao_charges(df_mol, dm_saiao):
    """IAO (SAIAO) based charges 

    Args:
        df_mol (pd.dataframe f): rom get_orbtypes_df(), has nuclear_charge column
        dm_saiao (np.array float64): SAIAO density matrix

    Returns:
        SAIAO charges: norb x norb
    """    
    df_mol['elec_density'] = np.diagonal(dm_saiao)
    df_mol['atm_charge_density'] = df_mol['nuclear_charge'].astype('float64') - df_mol.groupby(by = 'atm_id')['elec_density'].transform('sum')
    return np.diag(df_mol['atm_charge_density'])

def get_saiao_locality(mol, C_ao_saiao):
    """Boys localization feature for SAIAO: f_i = < i | r^2 | i> - <i | r | i> ^ 2

    Args:
        mol (pyscf.mol)
        C_ao_saiao (np.array float64): AO to SAIAO rotation

    Returns:
        boys: norb x norb matrix
    """    
    # 
    r1 = mol.intor_symmetric('int1e_r')
    r2 = mol.intor_symmetric('int1e_r2')
    r1_saiao = np.einsum('ui, xuv, vi -> xi', C_ao_saiao, r1, C_ao_saiao, optimize = True)
    r2_saiao = np.einsum('ui, uv, vi -> i', C_ao_saiao, r2, C_ao_saiao, optimize = True)
    boys = r2_saiao - np.einsum('xi, xi -> i', r1_saiao, r1_saiao, optimize = True)
    return np.diag(boys)
