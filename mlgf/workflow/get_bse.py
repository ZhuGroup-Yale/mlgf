import numpy as np
import argparse
import json
import psutil
import scipy

from pyscf import lib, dft, gto
from pyscf.lib import logger

from fcdmft.gw.mol.bse import BSE, _get_oscillator_strength
from fcdmft.df import addons

# TODO: import this from fcdmft when merged, remove this function
def cholesky_eri_mo(mol, mo_coeffs, auxbasis='weigend+etb', dataname='eri_mo',
                    lindep=1e-12, int2c='int2c2e', decompose_j2c='cd',
                    int3c='int3c2e', mosym='s1', comp=1,
                    max_memory=None, auxmol=None):
    
    from fcdmft.df.outcore import _guess_shell_ranges_L
    from pyscf import ao2mo

    assert (mosym in ('s1', 's2'))
    verbose = getattr(mol, 'verbose', 0)
    log = logger.new_logger(mol, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    time1 = time0
    assert comp == 1

    if max_memory is None:
        max_memory = mol.max_memory

    if not max_memory is None:
        assert (type(max_memory) is int or type(max_memory) is float)

    else:
        mem = psutil.virtual_memory()
        max_memory = int(0.8*mem.available / (1024 * 1024))

    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    t0 = (logger.process_clock(), logger.perf_counter())

    j2c = auxmol.intor(int2c, hermi=1)
    if decompose_j2c == 'eig':
        low = _eig_decompose(mol, j2c, lindep)
    else:
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
            decompose_j2c = 'cd'
        except scipy.linalg.LinAlgError:
            low = _eig_decompose(mol, j2c, lindep)
            decompose_j2c = 'eig'
    j2c = None
    naux, naoaux = low.shape
    log.debug('size of aux basis %d', naux)
    log.timer_debug1('2c2e', *t0)

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]
    naoaux = ao_loc[-1] - nao

    ijmosym, nij_pair, moij, ijshape = \
                ao2mo.incore._conc_mos(mo_coeffs[0], mo_coeffs[1],
                                       compact=(mosym=='s2'))


    nao_pair = nao * (nao+1) // 2
    buflen = max(int(max_memory*.24e6/8/nao_pair/comp), 1)
    shranges = _guess_shell_ranges_L(auxmol, nao_pair, buflen, 's2ij')
  

    log.debug1('shranges = %s', shranges)

    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    bufs1 = np.empty((comp*max([x[2] for x in shranges]), nao_pair))
    bufs2 = np.empty_like(bufs1)
    
    if comp == 1:
        dshape = (naoaux, nij_pair)
    else:
        dshape = (comp, naoaux, nij_pair)
    
    eri = np.empty(dshape)


    row = 0
    for istep, sh_range in enumerate(shranges):
        bufs2, bufs1 = bufs1, bufs2
        bstart, bend, nrow = sh_range
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+bstart, mol.nbas+bend)
        ints = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                       "s2ij", ao_loc, cintopt, out=bufs1)
        if ints.flags.f_contiguous:
            ints = ints.T
        if comp == 1:
            ao2mo._ao2mo.nr_e2(ints, moij, ijshape, aosym="s2kl", mosym=ijmosym, out=eri[row:row+nrow])
        else:
            ao2mo._ao2mo.nr_e2(ints.reshape(comp*nrow, nao_pair),
                               moij, ijshape, aosym="s2kl", mosym=ijmosym, out=eri[:,row:row+nrow].reshape((comp*nrow, nij_pair)))
        sh_range = shranges[istep]
        bstart, bend, nrow = sh_range
        row += nrow
        log.debug('int3c2e+MO [%d/%d], aux [%d:%d], nrow = %d',
                  istep+1, len(shranges), *sh_range)
        time1 = log.timer('gen mo eri [%d/%d]' % (istep+1,len(shranges)), *time1)

    log.debug(f'j2c-decompose {low.shape}, {eri.shape} -> {eri.shape}')

    if decompose_j2c == 'cd':
        assert eri.flags.c_contiguous
        trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, ints))
        dat = trsm(1.0, low, eri.T, lower=True, trans_a = 1, side = 1, overwrite_b=True).T
        cderi = dat
    else:
        ncolmax = bufs1.size // (naoaux*comp)
        for icol0, icol1 in lib.prange(0, nij_pair, ncolmax):
            cderi_block = np.matmul(low, eri[:,icol0:icol1], out=bufs1)
            eri[:naux, icol0:icol1] = cderi_block
        eri.resize(naux, nij_pair)
        cderi = eri
    log.timer('j2c-decompose', *time1)
    log.timer('total time', *time0)
    return cderi

def get_active_space_Lpq(mf, nocc_act, nvir_act):

    nmo = len(mf.mo_energy)
    nocc = mf.mol.nelectron // 2
    nvir = nmo - nocc

    nocc_act = nocc if nocc_act is None else min(nocc, nocc_act)
    nvir_act = nvir if nvir_act is None else min(nvir, nvir_act)
    nmo_act = nocc_act + nvir_act
    
    arg_mo_coeff = mf.mo_coeff.copy()[:,(nocc-nocc_act):(nocc+nvir_act)]
    arg_mo_coeff = (arg_mo_coeff, arg_mo_coeff)

    Lpq_mo_cd = cholesky_eri_mo(mf.mol, arg_mo_coeff, auxbasis = f"{mf.mol.basis}-ri", mosym = 's1')
    Lpq_mo_cd = Lpq_mo_cd.reshape((Lpq_mo_cd.shape[0], nocc_act+nvir_act, nocc_act+nvir_act))

    return Lpq_mo_cd

# TODO: interface this correctly with new fcdmft
def get_bse_singlets(mlf_chkfile, qpe, nmo, nocc, nroot = 1, bse_active_space = None, xc = 'pbe0'):
    """generates singlet excited state from qpe and existing scf stored in mlf_chkfile 

    Args:
        mlf_chkfile (string): chkfile with scf and mol objects
        qpe (np.float64): quasiparticle energies (true or ML)
        nmo (float64): number of MO
        nocc (float64): number of occupied MOs
        nroot (int, optional): number of BSE excited states to get. Defaults to 20.
        bse_active_space (list, optional): if two floats, does BSE for QPE this energy window, if integers, does BSE in the active space of bse_active_space[0] occupied and bse_active_space[1] virtuals. Defaults to None, where BSE is done in the full space.
        xc (str, optional): DFT functional. Defaults to 'pbe0'.

    Returns:
        _type_: _description_
    """    

    scf_data = lib.chkfile.load(mlf_chkfile, 'scf')
    mol = lib.chkfile.load_mol(mlf_chkfile)

    if bse_active_space is not None:
        if type(bse_active_space[0]) is float:
            top_mo = qpe[nocc] + bse_active_space[1]
            nocc_act = nocc - sum(qpe < bse_active_space[0])

            nvir_act = nmo - nocc - sum(qpe > top_mo)
            print(f"BSE (nocc, nvirt) from {bse_active_space[0]} to {top_mo} Hartree: ", nocc_act, nvir_act, flush = True)
            bse_active_space = [nocc_act, nvir_act]
            
        else:
            nocc_act = min(bse_active_space[0], nocc)
            nvir_act = min(bse_active_space[1], nmo - nocc)

    else:
        nocc_act = nocc
        nvir_act = nmo - nocc
    
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.__dict__.update(scf_data)
    mf.mol.max_memory=350000

    # get the density fitting integral Lpq for truncated BSE
    Lpq = get_active_space_Lpq(mf, nocc_act, nvir_act)
    
    # initialize BSE object in active space
    qpe_act = qpe[(nocc-nocc_act):(nocc+nvir_act)]
    mybse = BSE(nocc=nocc_act, mo_energy=qpe_act, Lpq=Lpq, verbose = mf.mol.verbose, TDA = True)

    logger.info(mybse, f"BSE active space (nocc, nvirt): {nocc_act}, {nvir_act}")

    # calculate lowest nroot singlet excited states
    mybse.nroot = nroot
    exci_s = mybse.kernel('s')[0]

    # calculate BSE oscillator strength in the subspace
    X_vec = np.zeros((nroot, nocc, nmo - nocc ))
    Y_vec = np.zeros((nroot, nocc, nmo - nocc))

    start_idx = nocc - nocc_act
    X_vec[:,start_idx:,:nvir_act] = mybse.X_vec[0]
    Y_vec[:,start_idx:,:nvir_act] = mybse.Y_vec[0]

    dipole, oscillator_strength = _get_oscillator_strength(multi=mybse.multi, exci=mybse.exci, X_vec=[X_vec], Y_vec=[Y_vec], mo_coeff=mf.mo_coeff[np.newaxis, ...], nocc=[nocc], mol=mf.mol)
    
    return exci_s, dipole, oscillator_strength, bse_active_space

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='get_bse.py')
    # must have key jobs with a list of dictionaries, each with key model_file, output_file, mlf_chkfile
    parser.add_argument('--json_spec', required=True, help='head directory where all molecules subdirectories have checkpoints from generate.py')
    # by default True

    args = parser.parse_args()
    json_spec = args.json_spec

    defaults = {'bse_nroot' : 1, 'bse_active_space' : None}

    assert('.json' in json_spec)

    with open(json_spec) as f:
        spec = json.load(f)
    
    bse_active_space = spec.get('bse_active_space', None)
    bse_nroot = spec.get('bse_nroot', defaults['bse_nroot'])
    validation_files = spec['validation_files']
    validation_keys = spec['validation_keys']
    skip_true_files = spec.get('skip_true_files', [])
    for validation_file in validation_files:
        for validation_key in validation_keys:
            validation = lib.chkfile.load(validation_file, validation_key)
            mlf_chkfile = validation['mlf_chkfile']
            if type(mlf_chkfile) is bytes:
                mlf_chkfile = mlf_chkfile.decode('utf-8')
            nmo = len(validation['mo_energy'])
            nocc = validation['nocc']

            qpe_ml = validation['qpe_ml']
            exci_s_ml, dipole_ml, oscillator_strength_ml, bse_active_space_ml = get_bse_singlets(mlf_chkfile, qpe_ml, nmo, nocc, nroot = bse_nroot, bse_active_space = bse_active_space, xc = 'pbe0')
            validation[f'exci_s_ml_nroot{bse_nroot}'] = exci_s_ml
            validation[f'dipole_ml_nroot{bse_nroot}'] = dipole_ml
            validation[f'oscillator_strength_ml_nroot{bse_nroot}'] = oscillator_strength_ml
            
            if validation_file in skip_true_files:
                lib.chkfile.save(validation_file, validation_key, validation)
                continue
            
            try:

                qpe_true = validation['qpe_true']
                exci_s_true, dipole_true, oscillator_strength_true, _ = get_bse_singlets(mlf_chkfile, qpe_true, nmo, nocc, nroot = bse_nroot, bse_active_space = bse_active_space_ml, xc = 'pbe0')

                validation[f'exci_s_true_nroot{bse_nroot}'] = exci_s_true
                validation[f'dipole_true_nroot{bse_nroot}'] = dipole_true
                validation[f'oscillator_strength_true_nroot{bse_nroot}'] = oscillator_strength_true
            except KeyError as e:
                print(validation_file, validation_key, ' has keyerror: ', str(e))
            
            lib.chkfile.save(validation_file, validation_key, validation)
        
 
