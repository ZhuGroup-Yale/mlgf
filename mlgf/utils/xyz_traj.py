import sys
import numpy as np
import pyscf.data.elements as elements
import pyscf.gto.mole
import json
import io

class SmallMolecule:
    def __init__(self, atom_charges, elements, atom_coords):
        self.atom_charges = np.array(atom_charges)
        self.elements = elements
        # row = atom, col = x,y,z
        self.atom_coords = np.array(atom_coords)
    def __len__(self):
        return len(self.elements)
    
    @staticmethod
    def from_atomlist(atomlist):
        return SmallMolecule([int(elements.charge(x[0])) for x in atomlist], \
                            [x[0] for x in atomlist], \
                        [x[1] for x in atomlist])
    
    @staticmethod
    def from_dict(d):
        return SmallMolecule(d['atom_charges'], d['elements'], d['atom_coords'])
    def get_atomlist(self):
        return [[self.elements[i], tuple(self.atom_coords[i,:].flatten())] for i in range(len(self))]

    def __iter__(self):
        return iter(self.get_atomlist())
    
    def __str__(self):
        return "".join('  {:<6}{:> 25.17f}{:> 25.17f}{:> 25.17f}\n'.format(atom[0], *atom[1]) for atom in self.get_atomlist())
    
    __repr__ = __str__
    @staticmethod
    def from_pyscf_mol(mol):
        if not mol._built:
            mol.build()
        syms = [tup[0] for tup in mol._atom]
        return SmallMolecule(mol.atom_charges(), syms, mol.atom_coords(unit='A'))

    def perturb_coords(self, perturbation):
        return SmallMolecule(self.atom_charges, self.elements, self.atom_coords + perturbation)

    def __eq__(self, other):
        return np.all(np.equal(self.atom_charges, other.atom_charges)) and \
            self.elements == other.elements and \
            np.allclose(self.atom_coords, other.atom_coords)

class MolEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SmallMolecule):
            return obj.get_atomlist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_mol_list(mol_list, fp, **kwargs):
    json.dump(mol_list, fp, cls=MolEncoder, indent=2, sort_keys=True, **kwargs)

def get_mol_json(mol_list, **kwargs):
    return json.dumps(mol_list, cls=MolEncoder, indent=2, sort_keys=True, **kwargs)

def load_mol_list(fp):
    return [SmallMolecule.from_atomlist(m) for m in json.load(fp)]

def load_mols_json(s):
    return [SmallMolecule.from_atomlist(m) for m in json.loads(s)]

def read_xyz(f):
    if not hasattr(f, 'readline'):
        raise TypeError("Expected file-like object with readline method")
    traj = []
    comments = []
    while True:
            #traj.append(xyz_read_frame(f))
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError as e:
                if f.readline() == '':
                    # end of file
                    break
                else:
                    raise IOError("Expected number of atoms in first line of xyz frame, got:\n" + line) from e

            comments.append(f.readline())
            atoms = []
            for i in range(natoms):
                line = f.readline()
                try:
                    atom, x, y, z = line.split()
                    atoms.append((atom, (float(x), float(y), float(z))))
                except ValueError as e:
                    raise IOError("Expected atomsym X Y Z, got:\n" + line) from e
            traj.append(atoms)
    return traj

def read_xyzfile(filename):
    with open(filename) as f:
        return read_xyz(f)
    

def dump_xyz(f, *args, **kwargs):
    if isinstance(f, io.IOBase):
        _dump_xyz(f, *args, **kwargs)
    else:
        with open(f, 'w') as fp:
            _dump_xyz(fp, *args, **kwargs)

def dumps_xyz(molecules, **kwargs):
    with io.StringIO() as sio:
        _dump_xyz(sio, molecules, **kwargs)
        xyzstring = sio.getvalue()
    return xyzstring
 
def _dump_xyz(fp, molecules, comments = None):
    if isinstance(molecules, SmallMolecule):
        molecules = [molecules]
    for idx, m in enumerate(molecules):
        if isinstance(m, pyscf.gto.mole.Mole):
            m = SmallMolecule.from_pyscf_mol(m)
        comment = None if comments is None else comments[idx]
        _write_xyz_frame(fp, m, idx, comment=comment)

def _write_xyz_frame(fp, molecule, idx, comment=None):
    fp.write(f'{len(molecule)}\n')
    if comment is None:
        comment = f' i = {idx}'
    fp.write(f'{comment}\n')
    for atom in molecule:
        fp.write('  {:<6}{:> 25.17f}{:> 25.17f}{:> 25.17f}\n'.format(atom[0], *atom[1]))
    

def water_get_geometry(mol):
    chgs = np.array(mol.atom_charges)
    coords = np.array(mol.atom_coords)
    where_oxy = np.flatnonzero(chgs == 8)[0]
    hydrogens = np.flatnonzero(chgs == 1)
    oxy_coord = coords[where_oxy,:]
    hyd_lengths = np.linalg.norm(coords[hydrogens] - oxy_coord, axis=1).tolist()
    cos_angle = np.dot(coords[hydrogens[0],:] - oxy_coord, coords[hydrogens[1],:] - oxy_coord) / (hyd_lengths[0] * hyd_lengths[1])
    return hyd_lengths, np.arccos(cos_angle)


if __name__ == '__main__':
    traj = read_xyzfile(sys.argv[1])
    lengths = []
    angles = []
    for frame in traj:
        mole = SmallMolecule.from_atomlist(frame)
        hyd_lengths, bond_angle = water_get_geometry(mole)
        lengths.append(hyd_lengths)
        angles.append(bond_angle)
    import IPython; IPython.embed()