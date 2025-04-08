import argparse
import numpy as np
import pyscf
from mlgf.lo.saao import get_C_ao_iao, get_saao
import IPython

def get_saiao(mol, mf):    
    dm = mf.make_rdm1()
    S_ao = mf.get_ovlp()
    
    C_ao_iao = get_C_ao_iao(mf, minao='minao')
    CS = np.dot(C_ao_iao.T, S_ao)
    dm_iao = CS @ dm @ CS.T
    C_iao_saiao = get_saao(mol, dm_iao)
    
    testv = C_iao_saiao.T @ C_iao_saiao
    assert(np.linalg.norm(testv-np.eye(testv.shape[0]))<1e-8)
    return C_ao_iao, C_iao_saiao

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='saiao_preplot.py')
    parser.add_argument('mol', metavar='MOLECULE', help='xyz file')
    parser.add_argument('basis', metavar='BASIS', help='basis set')
    parser.add_argument('--cart', action='store_true', default=False, help='use cartesian basis')
    
    args = parser.parse_args()
    
    mol = pyscf.gto.M(atom=args.mol, basis=args.basis, cart=args.cart, symmetry=False)
    mol.build()
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    
    
    C_ao_iao, C_iao_saiao = get_saiao(mol, mf)
    C_ao_saiao = C_ao_iao @ C_iao_saiao
    
    mol_cart = pyscf.gto.M(atom=args.mol, basis=args.basis, cart=True)
    overlap = mol_cart.intor_symmetric('cint1e_ovlp_cart')
    for orb in range(C_ao_saiao.shape[1]):
        wf_ao = C_ao_saiao[:,orb]
        wf_ao = wf_ao[:,np.newaxis]
        wfaocart = pyscf.scf.addons.project_mo_nr2nr(mol, wf_ao, mol_cart)
        wfaocart = wfaocart / np.sqrt(wfaocart.T @ overlap @ wfaocart)
        np.save(f'wf_saiao_{orb}.npy', wfaocart[:])
        
        wf_ao = C_ao_iao[:,orb]
        wf_ao = wf_ao[:,np.newaxis]
        wfaocart = pyscf.scf.addons.project_mo_nr2nr(mol, wf_ao, mol_cart)
        wfaocart = wfaocart / np.sqrt(wfaocart.T @ overlap @ wfaocart)
        np.save(f'wf_iao_{orb}.npy', wfaocart[:])
        
    dens = mf.make_rdm1()
    
    dens_cart = pyscf.scf.addons.project_dm_nr2nr(mol, dens, mol_cart)
   

    for i in range(2):
        wfaocart = np.zeros_like(wfaocart)
        wfaocart[i] = 1
        wfaocart = wfaocart / np.sqrt(wfaocart.T @ overlap @ wfaocart)
        np.save(f'mo_{i}.npy', wfaocart[:])

    np.save('dens.npy', dens_cart)
    
    #print(np.diagonal(overlap))