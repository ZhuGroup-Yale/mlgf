import argparse
import numpy as np
import math
import itertools
from mlgf.utils.xyz_traj import dump_xyz
import sys

def get_atomlist(ang, b1, b2, elements=["O", "H", "H"]):
    return [
        [elements[0], (0, 0, 0)],
        [elements[1], (b1, 0, 0)],
        [elements[2], (b2 * math.cos(ang), b2 * math.sin(ang), 0)],
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="triatomic_geom_grid.py")

    parser.add_argument("--ang-pts", metavar="N", type=int, default=30, required=True)
    parser.add_argument(
        "--bond-pts",
        metavar="EXPR",
        type=str,
        required=True,
        help='e.g. "np.linspace(0.5, 1.1, 30)"',
    )
    parser.add_argument(
        "--elements",
        metavar="X",
        type=str,
        nargs=3,
        default=["O", "H", "H"],
        help="e.g. O H H",
    )
    args = parser.parse_args()

    ang_pts = np.linspace(0, np.pi, args.ang_pts + 1)
    ang_pts = ang_pts[1:]

    bond_pts = eval(args.bond_pts)

    inds = np.tril_indices(len(bond_pts), k=0)

    bond_pairs = np.column_stack((bond_pts[inds[0]], bond_pts[inds[1]]))

    mols = [
        get_atomlist(ang, *bp, elements=args.elements)
        for ang, bp in itertools.product(ang_pts, bond_pairs)
    ]

    dump_xyz(sys.stdout, mols)
