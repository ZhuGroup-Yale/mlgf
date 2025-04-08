from mlgf.utils.xyz_traj import SmallMolecule, read_xyzfile, water_get_geometry
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import IPython

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='water_geom_kdeplot.py')
    parser.add_argument('xyzfile', type=str, help='xyz file')
    parser.add_argument('--title', type=str, help='title for plot', default="Water geometry")
    parser.add_argument('--corr', action='store_true', help='plot correlation between bond lengths and angles')
    parser.add_argument('--embed', action='store_true', help='embed in IPython shell after reading xyz file')
    args = parser.parse_args()
    traj = read_xyzfile(args.xyzfile)
    lengths = []
    angles = []
    for frame in traj:
        mole = SmallMolecule.from_atomlist(frame)
        hyd_lengths, bond_angle = water_get_geometry(mole)
        lengths.append(hyd_lengths)
        angles.append(bond_angle)
    angles = np.degrees(np.array(angles))
    lengths = np.array(lengths)
    if args.embed:
        IPython.embed()
    elif args.corr:
        df = pd.DataFrame(lengths)
        df.columns = ['Bond 1', 'Bond 2']
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x='Bond 1', y='Bond 2', ax=ax)
        plt.show()
    else:
        l0df = pd.DataFrame(np.column_stack((lengths[:,0],angles)))
        l1df = pd.DataFrame(np.column_stack((lengths[:,1],angles)))
        l0df['bond'] = "Bond 1"
        l1df['bond'] = "Bond 2"
        df = pd.concat((l0df,l1df))
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=0, y=1, hue='bond', ax=ax)
        ax.set_xlabel("O-H bond length (angstroms)")
        ax.set_ylabel("O-H bond angle (deg.)")
        ax.title.set_text(args.title)
        plt.show()