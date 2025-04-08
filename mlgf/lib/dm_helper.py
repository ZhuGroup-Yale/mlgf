from pyscf import lib
from mlgf.lib.dos_helper import get_g0
import numpy as np
from pyscf import data

einsum = lib.einsum

def dm_mo_to_ao(dm_mo, coeff):
    """Rotate MO dm to AO

    Parameters
    ----------
    dm_mo : float64 2d array
        MO
    coeff : float64 2d array
        MO coefficients

    Returns
    -------
    float64 2d array
        AO rdm1
    """    
    return np.dot(coeff, dm_mo).dot(coeff.T)

def make_frozen_no(dm, mo_energy, mo_coeff, nocc, thresh=1e-6, pct_occ=None, nvir_act=None):
    """Frozen natural orbitals

    Parameters
    ----------
    dm : float64 2d array
        MO density matrix
    mo_energy : float64 1d array
        MO energy
    mo_coeff : float64 2d array
        MO coefficients
    nocc : int
        number of docc orbitals
    thresh : float, optional
        Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
    pct_occ : float, optional
        Percentage of total occupation number. Default is None. If present, overrides `thresh`.
    nvir_act : int, optional
        Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.

    Returns
    -------
    frozen : list or ndarray
            List of orbitals to freeze
    no_coeff : ndarray
        Semicanonical NO coefficients in the AO basis
    """    

    nmo = len(mo_coeff)

    n,v = np.linalg.eigh(dm[nocc:,nocc:])
    idx = np.argsort(n)[::-1]
    n,v = n[idx], v[:,idx]

    # print(n, flush = True)
    # print ('virtual NO occ:',n)

    if nvir_act is None:
        if pct_occ is None:
            nvir_act = np.count_nonzero(n>thresh)
        else:
            print(np.cumsum(n/np.sum(n)))
            nvir_act = np.count_nonzero(np.cumsum(n/np.sum(n))<pct_occ)

    fvv = np.diag(mo_energy[nocc:])
    fvv_no = np.dot(v.T, np.dot(fvv, v))
    # print(fvv.shape, fvv_no.shape)
    _, v_canon = np.linalg.eigh(fvv_no[:nvir_act,:nvir_act])

    no_coeff_1 = np.dot(mo_coeff[:,nocc:], np.dot(v[:,:nvir_act], v_canon))
    no_coeff_2 = np.dot(mo_coeff[:,nocc:], v[:,nvir_act:])
    no_coeff = np.concatenate((mo_coeff[:,:nocc], no_coeff_1, no_coeff_2), axis=1)

    return np.arange(nocc+nvir_act,nmo), no_coeff

# do not use mol.dip_moment, its slow
def get_dipole(mf, dm_ao):
    """dipole

    Parameters
    ----------
    mf : pyscf mf object
    dm_ao : float64 2d array
        AO rdm1

    Returns
    -------
    float
        dipole
    """    
    return np.linalg.norm(mf.dip_moment(dm = dm_ao, verbose = 0))

def traceless_quadrupole(mol, dm, units="AU", charge_center_origin=True):
    """Calculate the traceless quadrupole moment of a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        molecule
    dm : array_like
        density matrix
    units : str, optional
        Specifies the units of the quadrupole moment.
        Must be "SI" or "AU" (default).
    charge_center_origin : bool, optional
        Specifies whether the center of nuclear charge
        should be used as the origin.

    Returns
    -------
    Q : (3, 3) ndarray
        Traceless quadrupole moment tensor.

    Raises
    ------
    ValueError
        If `units` is not "SI" or "AU".

    """
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    if charge_center_origin:
        charge_center = coords.T @ charges / charges.sum()
    else:
        charge_center = np.zeros(3, dtype=float)
    with mol.with_common_orig(charge_center):
        quad_ints = mol.intor_symmetric("int1e_rr", comp=9).reshape((3, 3, -1))
    r_nuc = coords - charge_center[None, :]
    elec_q = quad_ints @ dm.ravel()
    nuc_q = np.einsum("g,gx,gy->xy", charges, r_nuc, r_nuc)
    tot_q = (nuc_q - elec_q) / 2
    tot_q_traceless = 3 * tot_q - np.eye(3) * np.trace(tot_q)

    if units == "AU":
        return tot_q_traceless
    if units == "SI":
        quadrupole_au_to_si = data.nist.AU2DEBYE * data.nist.BOHR
        return tot_q_traceless * quadrupole_au_to_si

    msg = 'Units must be "SI" or "AU"'
    raise ValueError(msg)


def scalar_quadrupole(mol, dm, units="SI", charge_center_origin=True):
    """Calculate the scalar quadrupole moment of a molecule.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        molecule
    dm : array_like
        density matrix
    units : str, optional
        Specifies the units of the quadrupole moment.
        Must be "SI" or "AU" (default).
    charge_center_origin : bool, optional
        Specifies whether the center of nuclear charge
        should be used as the origin.
    
    Returns
    -------
    scalar_q : float
        Scalar quadrupole moment.

    """
    tot_q_traceless = traceless_quadrupole(
        mol, dm, units=units, charge_center_origin=charge_center_origin
    )
    evs, _ = np.linalg.eigh(tot_q_traceless)
    return (evs[-1] - evs[0]) / 2

def get_dm_linear(sigma, dm_freqs, dm_wts, mo_energy, vk_minus_vxc):
    """Get GW density matrix from G(it=0).
    G(it=0) = \int G(iw) dw
    As shown in doi.org/10.1021/acs.jctc.0c01264, calculate G0W0 Green's function using Dyson equation is not
    particle number conserving.
    The linear mode G = G0 + G0 Sigma G0 is particle number conserving.

    Parameters
    ----------
    sigma : complex128
        sigmaI
    dm_freqs : complex128
        imaginary frequencies that sigma is evaluated on
    dm_wts : float64
        Gauss-Legendre integration weights for dm_freqs
    mo_energy : float64
        DFT MO energies
    vk_minus_vxc : float64
        static self-energy from DFT/HF

    Returns
    -------
    rdm1 : double 2d array
        one-particle density matrix
    """    
    nw, nmo = sigma.shape[-1], len(mo_energy)
    
    gf0 = get_g0(dm_freqs, mo_energy, 0.)

    gf = np.zeros_like(gf0)
    for iw in range(gf.shape[-1]):
        gf[:,:,iw] = gf0[:,:,iw] + np.dot(gf0[:,:,iw], (vk_minus_vxc + sigma[:,:,iw])).dot(gf0[:,:,iw])
    
    # gf = gf0 + np.einsum('ijw,jkw,klw->ilw', gf0, vk_minus_vxc + sigma_dm_grid, gf0, optimize = True) # slow
    rdm1 = 2./np.pi * einsum('ijw,w->ij',gf,dm_wts) + np.eye(nmo)
    return rdm1.real

def get_dm_dyson(sigma, dm_freqs, dm_wts, mo_energy, vk_minus_vxc):
    """Get CC density matrix from G(it=0) with full Dyson equation.
    G(it=0) = \int G(iw) dw

    Parameters
    ----------
    sigma : complex128
        sigmaI
    dm_freqs : complex128
        imaginary frequencies that sigma is evaluated on
    dm_wts : float64
        Gauss-Legendre integration weights for dm_freqs
    mo_energy : float64
        DFT MO energies
    vk_minus_vxc : float64
        static self-energy from DFT/HF

    Returns
    -------
    rdm1 : double 2d array
        one-particle density matrix
    """   
    nw, nmo = dm_freqs.shape[0], len(mo_energy)
    
    gf0 = get_g0(dm_freqs, mo_energy, 0.)
    gf = np.linalg.inv(np.linalg.inv(gf0.T) - (np.expand_dims(vk_minus_vxc, -1) + sigma).T).T
    rdm1 = 2./np.pi * einsum('ijw,w->ij',gf,dm_wts) + np.eye(nmo)
    return rdm1.real
