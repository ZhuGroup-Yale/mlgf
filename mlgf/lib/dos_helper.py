import numpy as np

def get_g0(omega, mo_energy, eta):
    """Get non-interacting Green's function.

    Parameters
    ----------
    omega : double or complex array
        frequency grids
    mo_energy : double 1d array
        orbital energy
    eta : double
        broadening parameter

    Returns
    -------
    gf0 : complex 3d array
        non-interacting Green's function
    """
    nmo = len(mo_energy)
    nw = len(omega)
    gf0 = np.zeros(shape=[nmo, nmo, nw], dtype=np.complex128)
    gf0[np.arange(nmo), np.arange(nmo), :] = 1.0 / (omega[np.newaxis, :] + 1j * eta - mo_energy[:, np.newaxis])
    return gf0

def get_dos_hf(mo_energy, freqs, eta):
    GHFR = get_g0(freqs, mo_energy, eta)
    return -np.trace(GHFR.imag, axis1=0, axis2=1) / np.pi

def calc_dos(freqs, eta, acobj, mo_energy, vk_minus_vxc=None, diag=False):
    """Generate DOS using true sigma in the MO basis with optional full matrix inversion

    Parameters
    ----------
    freqs : array_like, shape (nw,), float
        real frequencies at which to evaluate the DOS
    eta : float
        broadening factor
    acobj : PadeAC, TwoPole, 
        fitted AC object, has ac_eval method
    mo_energy : array_like, shape (nmo,), float
        MO energies
    vk_minus_vxc : array_like, shape (nmo, nmo), float, optional
        Difference between HF exchange and DFT exchange-correlation potential
    diag : bool, optional
        If True, use diagonal approximation

    Returns
    -------
    array_like, shape (nw,), float
        density of states at the given frequencies
    """

    if not diag:
        GHFR = get_g0(freqs, mo_energy, eta)
        sigmaR = acobj.ac_eval(freqs + 1.j * eta)
        if vk_minus_vxc is not None:
            np.add(sigmaR, np.expand_dims(vk_minus_vxc, -1), out=sigmaR)
            # sigmaR += vk_minus_vxc
            
        # dyson equation to get G on the real axis
        GR = np.linalg.inv(np.linalg.inv(GHFR.T) - sigmaR.T).T

        # get DOS
        return -np.trace(GR.imag, axis1=0, axis2=1) / np.pi
    
    else: # diag
        # shape (nmo, nw)
        GHFR = np.reciprocal(np.add.outer(-mo_energy, freqs+1j*eta))

        # get sigma on the real axis
        sigmaR = acobj.diagonal().ac_eval(freqs+1.j*eta)

        if vk_minus_vxc is not None:
            sigmaR += np.expand_dims(np.diagonal(vk_minus_vxc), -1)
        
        # dyson equation
        GR = 1.0 / (1.0 / GHFR - sigmaR)

        # get DOS
        return -np.sum(GR.imag, axis=0) / np.pi
