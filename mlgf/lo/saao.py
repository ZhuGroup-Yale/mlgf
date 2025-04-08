import numpy as np

def get_saao(mol, dm, force_eigv_direction = False):
    '''
    Symmetry-adapted atomic orbitals
    Ref: JCP 153, 124111 (2020)

    Return:
        C_ao_saao : (nao, nsaao) 2D array, each column corresponds to a SAAO
    '''
    # count AOs on each nl (n: 0,1,2,...; l: s, p, d...)
    nao_nl = []
    for i in range(mol.nbas):
        for k in range(mol.bas_nctr(i)):
            nao_nl.append(2 * mol.bas_angular(i) + 1)

    # AO index on each nl
    idx_tmp_start = 0
    idx_nl = []
    for i in range(len(nao_nl)):
        idx_tmp_end = idx_tmp_start + nao_nl[i]
        idx_nl.append([*range(idx_tmp_start, idx_tmp_end)])
        idx_tmp_start = idx_tmp_end

    # diagonalize each nl block of mean-field density matrix
    C_ao_saao = np.zeros_like(dm)
    for i in range(len(nao_nl)):
        ix = np.ix_(idx_nl[i], idx_nl[i])
        e_nl, y_nl = np.linalg.eigh(dm[ix])
        if force_eigv_direction:
            slice_bool = np.sum(y_nl, axis = 0) < 0
            y_nl[:,slice_bool] = -y_nl[:,slice_bool]
        C_ao_saao[ix] = y_nl

    return C_ao_saao

def get_C_ao_iao(mf, minao, minao_val=None, minao_core=None):
    '''
    Intrinsic atomic orbital + projected atomic orbital (IAO+PAO) basis
    Ref: JCTC 9, 4834-4843 (2013)

    Return:
        C_ao_iao : (nao, niao) 2D array, each column corresponds to a IAO or PAO
                    Note: the orbital order is the same as original AO basis
    '''
    from libdmet.basis_transform import make_basis
    from pyscf.lo.iao import reference_mol

    if minao_val is None and minao_core is None:
        C_ao_iao = make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, tol = 1e-6)
        

        mol = mf.mol
        pmol = reference_mol(mol, minao=minao)
        B1_labels = mol.ao_labels()
        B2_labels = pmol.ao_labels()
        iao_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B2_labels)]
        virt_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label not in B2_labels)]
        iao_virt_idx = iao_idx + virt_idx

        C_ao_iao_ordered = np.zeros_like(C_ao_iao)
        for i, iorb in enumerate(iao_virt_idx):
            C_ao_iao_ordered[:,iorb] = C_ao_iao[:,i]
    else:
        assert (minao_val is not None)
        assert (minao_core is not None)
        from libdmet.lo import iao
        mol = mf.mol
        print('Building pmol_core and pmol_val', flush = True)
        pmol_core, pmol_val = iao.build_pmol_core_val(mol, minao_core, minao_val)
        C_ao_iao = make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, pmol_val=pmol_val,
                                                  pmol_core=pmol_core, tol = 1e-6)
        B1_labels = mol.ao_labels()
        B2_labels = pmol_core.ao_labels()
        B3_labels = pmol_val.ao_labels()
        core_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B2_labels)]
        val_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B3_labels)]
        virt_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label not in B2_labels and label not in B3_labels)]
        core_val_virt_idx = core_idx + val_idx + virt_idx

        C_ao_iao_ordered = np.zeros_like(C_ao_iao)
        for i, iorb in enumerate(core_val_virt_idx):
            C_ao_iao_ordered[:,iorb] = C_ao_iao[:,i]

    return C_ao_iao_ordered
