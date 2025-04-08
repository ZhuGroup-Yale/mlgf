import numpy as np

# tensors of dimension N_orb x Norb x freq_points
def unravel_rank3_sigma(gf_dyn):
    iu = np.triu_indices(gf_dyn.shape[0], k=1, m=None)
    return np.array([gf_dyn[:, :, m][iu] for m in range(gf_dyn.shape[-1])]).T


class BasisChanger:
    """Basis change helper class.
    
    Example:
        >>> from pyscf import gto, scf
        >>> import numpy as np
        >>> from mlgf.lo.saao import get_saao
        >>> from mlgf.utils.linalg import BasisChanger
        >>> mol = gto.M(atom="C 0 0 0; O 0 0 1.128", basis='ccpvdz', verbose=5)
        >>> mf = scf.RHF(mol)
        >>> mf.kernel()
        >>> C_ao_saao = get_saao(mol, mf.make_rdm1())
        >>> fock = mf.get_fock()
        >>> fock_saao1 = np.linalg.multi_dot((C_ao_saao.T, fock, C_ao_saao))
        >>> S_ao = mol.intor('int1e_ovlp')
        >>> ao2saao = BasisChanger(S_ao, C_ao_saao)
        >>> fock_saao2 = ao2saao.rotate_focklike(fock)
        >>> assert(np.allclose(fock_saao1, fock_saao2))
        >>>
        >>> SCS = np.linalg.multi_dot((np.linalg.inv(S_saao), C_ao_saao.T, S_ao))
        >>> dm_saao = reduce(np.dot, (SCS, dm, SCS.T))
        >>> dm_saao2 = ao2saao.rotate_denslike(dm)
        >>> assert(np.allclose(dm_saao, dm_saao2))
        
    """
    def __init__(self, S, C, to_orthonormal=False):
        """
        Construct an instance of the basis change helper class.

        Args:
            S (np.ndarray): overlap matrix in original basis
            C (np.ndarray): transformation matrix.
            
            The columns of C are the new basis vectors expressed in the original basis.
            
        """
        self.S = S
        self.C = C
        self.to_orthonormal = to_orthonormal
        
        if to_orthonormal:
            self.Cinv = np.dot(C.conj().T, S)
        else:
            self.Cinv = np.linalg.inv(C)
        
    def rotate_focklike(self, mat):
        """
        Transform a Fock-like matrix from the original basis to the new basis.
        This is applicable to Fock, Hcore, Vj, Vk, etc.

        Args:
            mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            r_mat: rotated matrix (or array)
        """
        if mat.ndim == 2:
            return np.linalg.multi_dot((self.C.conj().T, mat, self.C))
        elif mat.ndim > 2:
            return (self.C.conj().T @ mat.T @ self.C).T
        else:
            raise ValueError('mat.ndim must be >= 2')

    def rotate_denslike(self, mat):
        """
        Transform a density-like matrix from the original basis to the new basis.
        This is applicable to the density matrix, GF, etc.

        Args:
            mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            r_mat: rotated matrix (or array)
        """
        # for density matrix and GF
        if mat.ndim == 2:
            return np.linalg.multi_dot((self.Cinv, mat, self.Cinv.conj().T))
        elif mat.ndim > 2:
            return (self.Cinv @ mat.T @ self.Cinv.conj().T).T
        else:
            raise ValueError('mat.ndim must be >= 2')

    def rotate_oplike(self, mat):
        # May not be needed.
        if mat.ndim == 2:
            return np.linalg.multi_dot((self.Cinv, mat, self.C))
        elif mat.ndim > 2:
            return (self.Cinv @ mat.T @ self.C).T
        else:
            raise ValueError('mat.ndim must be >= 2')
    
    def rev_focklike(self, r_mat):
        """
        Transform a Fock-like matrix from the new basis to the original basis.
        This is applicable to Fock, Hcore, Vj, Vk, etc.

        Args:
            r_mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            mat: rotated matrix (or array)
        """
        # for Fock, Hcore, Vj, Vk, etc
        if r_mat.ndim == 2:
            return np.linalg.multi_dot((self.Cinv.conj().T, r_mat, self.Cinv))
        elif r_mat.ndim > 2:
            return (self.Cinv.conj().T @ r_mat.T @ self.Cinv).T
        else:
            raise ValueError('rmat.ndim must be >= 2')

    def rev_denslike(self, r_mat):
        """
        Transform a density-like matrix from the new basis to the original basis.
        This is applicable to the density matrix, GF, etc.

        Args:
            r_mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            mat: rotated matrix (or array)
        """
        # for density matrix and GF
        if r_mat.ndim == 2:
            return np.linalg.multi_dot((self.C, r_mat, self.C.conj().T))
        elif r_mat.ndim > 2:
            return (self.C @ r_mat.T @ self.C.conj().T).T
        else:
            raise ValueError('rmat.ndim must be >= 2')

    def rev_oplike(self, r_mat):
        # May not be needed.
        if r_mat.ndim == 2:
            return np.linalg.multi_dot((self.C, r_mat, self.Cinv))
        elif r_mat.ndim > 2:
            return (self.C @ r_mat.T @ self.Cinv).T
        else:
            raise ValueError('rmat.ndim must be >= 2')
    
    
    
    def transform(self, mat, mat_type='focklike', rev=False):
        if not rev:
            if mat_type == 'focklike':
                return self.rotate_focklike(mat)
            elif mat_type == 'denslike':
                return self.rotate_denslike(mat)
            elif mat_type == 'oplike':
                return self.rotate_oplike(mat)
            else:
                raise ValueError(f'Unknown mat_type {mat_type}')
        else:
            if mat_type == 'focklike':
                return self.rev_focklike(mat)
            elif mat_type == 'denslike':
                return self.rev_denslike(mat)
            elif mat_type == 'oplike':
                return self.rev_oplike(mat)
            else:
                raise ValueError(f'Unknown mat_type {mat_type}')
        
    
    def inverse(self):
        if self.to_orthonormal:
            S_tilde = np.eye(self.S.shape[0])
        else:
            S_tilde = self.rotate_focklike(self.S)
        
        if np.linalg.norm(self.S - np.eye(self.S.shape[0])) < 1.0e-8:
            return BasisChanger(S_tilde, self.Cinv, to_orthonormal=True)
        return BasisChanger(S_tilde, self.Cinv)
    
    def chain(self, other):
        S = self.S
        C = self.C @ other.C
        return BasisChanger(S, C, to_orthonormal=other.to_orthonormal)
    


focklike = {'fock', 'hcore', 'vj', 'vk', 'ovlp', 'sigmaI', 'sigma_fit'}

denslike = {'dm_hf', 'dm', 'gGW_ao', 'gGW'}

mat_type_map = {}
for f in focklike:
    mat_type_map[f] = 'focklike'
for d in denslike:
    mat_type_map[d] = 'denslike'
