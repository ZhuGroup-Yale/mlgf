import os
import numpy as np
import time
import argparse
import json
from functools import reduce

from pyscf import gto, scf, fci, lib, cc
from pyscf.lib import logger, temporary_env

try:
    from fcdmft.solver import fcigf, mpiccgf_mor
except ModuleNotFoundError:
    pass

from fcdmft.gw.mol.gw_ac import GWAC, _get_scaled_legendre_roots, \
    get_g0, get_sigma, get_sigma_outcore

from mlgf.lib.ml_helper import get_pade18

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
except ModuleNotFoundError:
    rank = 0
    size = 1
    comm = None
    pass

def split_xyz(xyzfile):
    """split xyz file into atom coordinates

    Args:
        xyzfile (str): xyzfile

    Returns:
        str: formatted atom coords string for pyscf mol constructor
    """    
    with open(xyzfile, 'r') as f:
        s = f.read()
    s = s.split('\n')[2:]
    return  ';'.join(s).replace('*^', 'e')

def fast_ao_fock(mo_energy, mo_coeff, ovlp):
    C_ao_mo_f = reduce(np.matmul, (ovlp, mo_coeff))
    return reduce(np.matmul, (C_ao_mo_f, np.diag(mo_energy), C_ao_mo_f.T)) # Fock matrix in AO basis from diagonal MO basis

def do_rks_calculation(mol, chkfile, **kwargs):
    """
    do DFT calculation
    
    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        **kwawrgs (dict) : calculation parameters.

    Returns:
        dict: dictionary with electronic structure data
    """   

    # Assign variables from kwargs with default values
    xc = kwargs.get('xc', 'hf')
    init_guess = kwargs.get('init_guess', None)
    diis_start_cycle = kwargs.get('diis_start_cycle', None)
    dm_init = kwargs.get('dm_init', None)
    conv_tol = kwargs.get('conv_tol', None)

    if os.path.isfile(chkfile):
        mlf = lib.chkfile.load(chkfile, 'mlf')
        if mlf is None:
            mlf = {}
    else:
        mlf = {}
    
    # Hartree-Fock calculation
    mf = scf.RKS(mol)
    mf.chkfile = chkfile
    mf.xc = xc

    for key, val in kwargs.items():
        if key == 'dm_init':
            continue
        if val is not None:
            setattr(mf, key, val)
    
    mf.kernel(dm = dm_init)

    # DFT/HF calculation outputs, cheap stuff below
    mlf['e_mf'] = mf.e_tot                         # DFT energy    
    nocc =  mol.nelectron // 2           
    mlf['nocc'] = nocc                             # occupation number/number of electrons
    mlf['mo_occ'] = np.asarray(mf.mo_occ)          # occupation number of each orbital
    mlf['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mlf['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mlf['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mlf['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
    mlf['dm_hf'] = np.asarray(mf.make_rdm1())      # mean field density matrix

    # Fock matrix computed with calling mf.get_fock() which calls mf.get_veff()
    mlf['fock'] = np.asarray(fast_ao_fock(mlf['mo_energy'], mlf['mo_coeff'], mlf['ovlp']))   

    # More expesnive stuff (K matrix computed 2x, J matrix computed 1x)
    vj, vk = mf.get_jk()
    mlf['vj'] = np.asarray(vj)                     # Coulomb matrix
    mlf['vk'] = np.asarray(vk)                     # exchange matrix
    mlf['vxc'] = np.asarray(mf.get_veff() - vj)    # exchange-correlation matrix

    # the definition of the hamiltonian is a bit tricky here, need to multiply by -0.5 to get the correct definition of vk
    mlf['vk_hf'] = -0.5*np.asarray(vk)

    mlf['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    mlf['xc'] = xc

    return mf, mlf

def do_hf_calculation(mol, chkfile, **kwargs):
    """do HF calculation

    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        kwargs (dict): dictionary with calculation parameters

    Returns:
        dict: dictionary with electronic structure data
    """  

    # Assign variables from kwargs with default values
    init_guess = kwargs.get('init_guess', 'minao')
    diis_start_cycle = kwargs.get('diis_start_cycle', 1)
    dm_init = kwargs.get('dm_init', None)
    conv_tol = kwargs.get('conv_tol', 1e-9)

    # Hartree-Fock calculation
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.init_guess = init_guess
    mf.diis_start_cycle = diis_start_cycle
    mf.conv_tol = conv_tol
    mf.kernel(dm = dm_init)
    mf.kernel()

    # DFT/HF calculation outputs
    if os.path.isfile(chkfile):
        mlf = lib.chkfile.load(chkfile, 'mlf')
        if mlf is None:
            mlf = {}
    else:
        mlf = {}

    mlf['e_mf'] = mf.e_tot  # HF energy

    # DFT/HF calculation outputs, cheap stuff below
    nocc =  mol.nelectron // 2 # occupation number/number of electrons
    mlf['nocc'] = nocc
    mlf['xc'] = 'hf'
    mlf['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    mlf['mo_occ'] = np.asarray(mf.mo_occ)
    mlf['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mlf['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mlf['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mlf['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
    mlf['dm_hf'] = np.asarray(mf.make_rdm1())      # mean field density matrix

    # Fock matrix computed with calling mf.get_fock() which calls mf.get_veff()
    mlf['fock'] = np.asarray(fast_ao_fock(mlf['mo_energy'], mlf['mo_coeff'], mlf['ovlp']))   
    
    # More expesnive stuff (K matrix computed 1x, J matrix computed 1x)
    vj, vk = mf.get_jk()
    mlf['vj'] = np.asarray(vj)                     # Coulomb matrix
    mlf['vk'] = np.asarray(vk)                     # exchange matrix

    mlf['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0

    return mf, mlf

# xc is ignored
def do_fcigf_calculation(mol, chkfile,
                         mol_dict=None, gmres_tol=1e-6, eta=0, xc = 'hf'):
    """do FCI calculation

    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        mol_dict (dict): dictionary with electronic structure data
        gmres_tol (float): GMRES tol for linear solver to get FCIGF, defaults to 1e-6.
        eta (float): band-broadening
        xc (str, optional): dft functional. Defaults to 'hf'.

    Returns:
        dict: dictionary with electronic structure data
    """  
    if mol_dict is None:
        mol_dict = {}
    mf, mol_dict = do_hf_calculation(mol, chkfile, mol_dict=mol_dict)

    # FCI

    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]

    print('FCI energy = ', fci_energy)
    mol_dict['fci_energy'] = fci_energy

    dm_fci = cisolver.make_rdm1(cisolver.ci, mol.nao, mol.nelectron)
    nocc = mol.nelectron // 2
    ef = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    freqs, wts = get_pade18()
    omega = ef + 1j*freqs

    myfcigf = fcigf.FCIGF(cisolver, mf, tol=gmres_tol)

    orbs = range(len(mf.mo_energy))
    g_ip = myfcigf.ipfci_mo(orbs, orbs, omega.conj(), eta).conj()
    g_ea = myfcigf.eafci_mo(orbs, orbs, omega, eta)
    gf = g_ip + g_ea

    gf0 = get_g0(omega, mf.mo_energy, eta=0)
    sigmaI = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()

    for name, obj in zip(['ef', 'dm_fci', 'freqs', 'wts', 'sigmaI', 'omega_fit'],
                         [ef, dm_fci, freqs, wts, sigmaI, omega]):
        mol_dict[name] = np.asarray(obj)

    return mol_dict

def save_ccsd_chk(mycc, chkfile, log, purge_amplitudes = False):
    """save pyscf cc object with amplitudes needed for subsequent CCGF calculation

    Args:
        mycc (pyscf.cc)
        chkfile (_type_): pyscf/mlgf chkfile
        log (pyscf.lib.logger): pyscf logger object
        purge_amplitudes (bool, optional): purge the amplitudes before saving. Defaults to False.
    """    
    if purge_amplitudes:
        lib.chkfile.dump(chkfile, 'ccsd', {})
        return

    ccsd_dict = vars(mycc).copy()
    pop_keys = ['mol', '_scf', 'stdout','_nmo', '_nocc', 'callback', 'frozen']
    amplitude_keys = ['t1', 't2', 'l1', 'l2']
    if purge_amplitudes:
        pop_keys = pop_keys + amplitude_keys
    for key in pop_keys: 
        try:
            ccsd_dict.pop(key)
            if key in amplitude_keys:
                log.info(f'cluster amplitude {key} removed for memory saving.')
        except KeyError:
            pass
        
        if '_keys' in ccsd_dict.keys():
            try:
                ccsd_dict['_keys'].remove(key)
            except ValueError:
                pass

    # save to the same chk file as the scf under the new "ccsd" key
    lib.chkfile.dump(chkfile, 'ccsd', ccsd_dict)

# intended to be used like 
def do_ccsd_calculation(mol, chkfile, **kwargs):
    """do CCSD calculation and save amplitudes to file

    Args:
        mol (pyscf.mol)
        chkfile (_type_): mlgf/pyscf chkfile
        kwargs (dict, optional): calculation parameters

    Returns:
        dict: electronic structure data dict
    """    

    verbose = kwargs.get('verbose', 4)
    purge_amplitudes = kwargs.get('purge_amplitudes', False)
    
    mf, mlf = do_hf_calculation(mol, chkfile, **kwargs)
    mycc = cc.RCCSD(mf)

    log = logger.Logger(mol.stdout, verbose=verbose)
    # CCSD
    mycc = cc.RCCSD(mf)
    mycc.verbose = verbose
    log.info('Starting cc kernel...')
    mycc.kernel()
    log.info('Starting cc lambda solver...')
    mycc.solve_lambda()
    log.info('Getting cc density matrix...')
    dm_cc = mycc.make_rdm1()

    mlf['dm_cc'] = dm_cc
    mlf['e_mp2'] = mycc.emp2
    mlf['e_corr'] = mycc.e_corr
    save_ccsd_chk(mycc, chkfile, log, purge_amplitudes = purge_amplitudes)

    return mlf

# xc is ignored
def do_ccgfmpi_calculation(chkfile, **kwargs):
    """do ccgf with MPI parallelization over the orbitals

    Args:
        chkfile (chkfile): pyscf/mlf chkfile
    Returns:
        dict: mlf dictionary with CCGF data
    """    

    # Assign variables from kwargs with default values
    verbose = kwargs.get('verbose', 4)
    xc = kwargs.get('xc', 'hf')
    purge_amplitudes = kwargs.get('purge_amplitudes', True)
    nw = kwargs.get('nw', 30)
    gl_grid = kwargs.get('gl_grid', False)
    tol = kwargs.get('tol', 1e-4)

    # MPI comm, rank, size
    comm = kwargs.get('comm', None)
    rank = kwargs.get('rank', 0)
    size = kwargs.get('comm', 1)

    mol = lib.chkfile.load_mol(chkfile)
    mf = scf.RHF(mol)
    mf.xc = xc

    scf_data = lib.chkfile.load(chkfile, 'scf')
    mf.__dict__.update(scf_data)

    mycc = cc.RCCSD(mf)
    cc_data = lib.chkfile.load(chkfile, 'ccsd')
    mycc.__dict__.update(cc_data)

    log = logger.Logger(mol.stdout, verbose=verbose)

    # get ef and freqs (to be consistent with GW)
    nocc = mol.nelectron // 2
    ef = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0

    if not gl_grid:
        freqs, wts = get_pade18()
        omega = ef + 1j*freqs
    else:
        freqs, wts = _get_scaled_legendre_roots(nw=int(nw))
        omega = ef + 1j*freqs

    # run CCGF
    orbs = range(len(mf.mo_energy))
    log.info('Getting CCGF object from fcdmft...')
    myccgf = mpiccgf_mor.CCGF(mycc, tol=tol, verbose = verbose)
    myccgf.verbose = verbose
    
    log.info('Getting IP...')
    g_ip = myccgf.ipccsd_mo(orbs, orbs, omega.conj(), broadening=0).conj()
    log.info('Getting EA...')
    g_ea = myccgf.eaccsd_mo(orbs, orbs, omega, broadening=0)
    gf = g_ip + g_ea

    # get sigma(iw)
    gf0 = get_g0(omega, mf.mo_energy, eta=0)  
    sigmaI = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T.copy()

    # saving CCGF results
    for name, obj in zip(['ef', 'freqs', 'wts', 'sigmaI', 'omega_fit'],
                     [ef, freqs, wts, sigmaI, omega]):

        mlf[name] = np.asarray(obj)
    
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        save_ccsd_chk(mycc, chkfile, log, purge_amplitudes = purge_amplitudes)
    if comm is not None:
        comm.Barrier()

    mlf['gmres_tol'] = tol

    return mlf

def do_gwac_calculation(chkfile, **kwargs):
    """Do a G0W0 calculation with the scf_data in chkfile

    Args:
        mol (pyscf.mol):
        chkfile (chkfile): pyscf/mlf chkfile
        xc (str): DFT functional for pyscf
        outcore (bool, optional): split up GW calculation of rho response into a loop. Defaults to False.
        fullsigma (bool, optional): compute full nmo x nmo x nw sigma, otherwise only do diagonal (faster)
        nw (int, optional): nomega GL points to use for integration. Defaults to gw_ac default (100)
        nw2 (int, optional): nomega GL points on which to evaluate sigmaI; integration still is carried out on nw = 100. Defaults to None, in which case GWGF computes on the same 100 GL grid used for integration.
        orbs (list, optional): subset of sigmaI to compute, defaults to do all orbitals
    Returns:
        dict: mlf dictionary
    """    

    # Assign variables from kwargs with default values
    xc = kwargs.get('xc', 'hf')
    outcore = kwargs.get('outcore', False)
    fullsigma = kwargs.get('fullsigma', True)
    nw = kwargs.get('nw', 100)
    nw2 = kwargs.get('nw2', None)
    orbs = kwargs.get('orbs', None)
    frozen = kwargs.get('frozen', None)
    ac_iw_cutoff = kwargs.get('ac_iw_cutoff', 5.0)
    freqs = kwargs.get('freqs', None)
    wts = kwargs.get('wts', None)
    ac_idx = kwargs.get('ac_idx', None)
    if freqs is not None:
        freqs = np.asarray(freqs)
    if wts is not None:
        wts = np.asarray(wts)
    scf_data = lib.chkfile.load(chkfile, 'scf')
    mol = lib.chkfile.load_mol(chkfile)

    

    if scf_data is None:
        raise ValueError(f"No scf data found in file: {chkfile}")
    else:
        print(f'Found existing SCF calculation from {chkfile}')
        mf = scf.RKS(mol)
        mf.xc = xc
        mf.__dict__.update(scf_data)
        print('Done loading previous SCF!')
    
    if os.path.isfile(chkfile):
        mlf = lib.chkfile.load(chkfile, 'mlf')
        if mlf is None:
            mlf = {}
    else:
        mlf = {}

    time_init = time.time()

    # initialize gw_ac object
    gw = GWAC(mf)
    gw.ac = 'pade'
    gw.nw = nw
    gw.nw2 = nw2
    gw.ac_iw_cutoff = ac_iw_cutoff
    gw.frozen = frozen
    gw.orbs = orbs
    gw.verbose = 5
    gw.fullsigma = fullsigma
    gw.rdm = True
    gw.outcore = outcore
    gw.freqs = freqs # evaluations grid freqs
    gw.wts = wts # evaluation grid wts
    gw.ac_idx = ac_idx
    # compute sigmaI
    gw.kernel()

    # get ef
    ef = gw.ef
    
    time_gw = time.time() - time_init

    E_tot, E_hf, Ec = gw.energy_tot() # GW total energy, may be unstable

    dm_gw = gw.make_rdm1() # get GW density matrix
    
    # start from first nonzero frequency point as ML target, first point is 0.
    omega_fit = gw.freqs*1.0j + ef
    sigmaI = gw.sigmaI[:,:,1:]
    
    # GW part
    for name, obj in zip(['ef', 'dm_gw', 'freqs', 'wts', 'sigmaI', 'omega_fit', 'e_tot', 'e_hf', 'e_corr'],
                     [ef, dm_gw, gw.freqs, gw.wts, sigmaI, omega_fit, E_tot, E_hf, Ec]):
        mlf[name] = np.asarray(obj)

    # print(chkfile, np.abs(np.trace(dm_gw) - np.sum(mol_dict['mo_occ'])))
    mlf['time_gw'] = time_gw
    
    return mlf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='generate.py')
    parser.add_argument('--calc', required = True, default='dft+gwac', help='string specifying which calculation(s) to preform starting from xyz file. Multiple calculations are specified by seperating with + symbol')
    
    # system parameters are command line args
    parser.add_argument('--charge', required = False, default=0, help='system charge, assumed neutral')
    parser.add_argument('--basis', required = False, default='ccpvdz', help='')
    parser.add_argument('--xyz_file', required = False, default=None, help='xyz file to start DFT calculation from.')
    parser.add_argument('--chk_file', required = True, default=None, help='.chk file for reading and writing electronic structure data.')
    
    # calculation parameters stored in json
    parser.add_argument('--json_spec', required = False, default=None, help='json file holding keyword arguments for GW calculation')
    args = parser.parse_args()

    available_calc_types = [
        "dft+gwac", "dft", "gwac",
        "ccsd", "ccsd+ccgf",
        "fcigf",
    ]
    if args.json_spec is not None:
        assert('.json' in args.json_spec)
        with open(args.json_spec) as f:
            spec = json.load(f)
    else:
        spec = {}
    
    verbose = spec.get('verbose', 0)
    chkfile = args.chk_file
    xyzfile = args.xyz_file
    calculation_type = args.calc

    assert calculation_type in available_calc_types, f'--calc must be 1 of {available_calc_types}.'

    if calculation_type == 'dft':
        assert xyzfile is not None, 'Must supply an xyz_file for DFT calculations.'
        coords = split_xyz(xyzfile)                
        mol = gto.M(atom = coords, basis = args.basis, verbose=verbose, parse_arg=False, charge = args.charge)
        mf, mlf = do_rks_calculation(mol, chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)

    if calculation_type == 'gwac':
        mlf = do_gwac_calculation(chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)
    
    if calculation_type == 'dft+gwac':
        assert xyzfile is not None, 'Must supply an xyz_file for DFT calculations.'
        coords = split_xyz(xyzfile)                
        mol = gto.M(atom = coords, basis = args.basis, verbose=verbose, parse_arg=False, charge = args.charge)
        mf, mlf = do_rks_calculation(mol, chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)
        mlf = do_gwac_calculation(chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)
    
    if calculation_type == 'ccsd':
        assert xyzfile is not None, 'Must supply an xyz_file for CCSD calculations.'
        coords = split_xyz(xyzfile)                
        mol = gto.M(atom = coords, basis = args.basis, verbose=verbose, parse_arg=False, charge = args.charge)
        mlf = do_ccsd_calculation(mol, chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)
    
    if calculation_type == 'ccsd+ccgf':
        assert xyzfile is not None, 'Must supply an xyz_file for CCSD calculations.'
        if rank == 0:
            coords = split_xyz(xyzfile)                
            mol = gto.M(atom = coords, basis = args.basis, verbose=verbose, parse_arg=False, charge = args.charge)
            mlf = do_ccsd_calculation(mol, chkfile, **spec)
            lib.chkfile.save(chkfile, 'mlf', mlf)
        
        if comm is not None:
            comm.Barrier()
        
        spec['rank'] = rank
        spec['comm'] = comm
        spec['size'] = size

        mlf = do_ccgfmpi_calculation(chkfile, **spec)
        if rank == 0:
            lib.chkfile.save(chkfile, 'mlf', mlf)

    if calculation_type == 'fcigf':
        raise NotImplementedError



            


    


