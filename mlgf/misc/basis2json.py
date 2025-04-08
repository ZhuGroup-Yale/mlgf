#!/usr/bin/env python3

if __name__ == '__main__':
    import argparse
    import pyscf
    import json
    parser = argparse.ArgumentParser(description='Do an SCF calculation using pyscf and save the density matrix and basis set information.')
    parser.add_argument('molecule', metavar='MOLECULE', help='xyz file')
    parser.add_argument('basis', metavar='BASIS', help='basis set')
    parser.add_argument('-o', '--output', metavar='OUTPUT', help='output file stem (json and npy)')

    args = parser.parse_args()

    mol = pyscf.gto.M(atom=args.molecule, basis=args.basis, cart=True, symmetry=False)
    mol.build()
    
    
    nshell = mol.nbas
    nprimitive = sum(row[2] for row in mol._bas)
    nbf = sum(row[2] for row in mol._bas)
    natoms = mol.natm
    


    atom_coords = [tuple(mol.atom_coord(i)) for i in range(natoms)]
    atom_Z = [int(mol.atom_charge(i)) for i in range(natoms)]
    shells = []
    
    for shell in range(nshell):
        num_contracted = mol.bas_nctr(shell)
        for c in range(num_contracted):
            shell_dict = {
                'coords': tuple(mol.bas_coord(shell)),
                'Z': atom_Z[mol.bas_atom(shell)],
                'am': int(mol.bas_angular(shell)),
                'nprim': int(mol.bas_nprim(shell)),
                'coefs': list(mol.bas_ctr_coeff(shell)[:,c]),
                'original_coefs': list(mol.bas_ctr_coeff(shell)[:,c]),
                'exps': list(mol.bas_exp(shell)),
            }
            shells.append(shell_dict)

    print(json.dumps(shells, indent=1))