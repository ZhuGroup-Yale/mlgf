from mlgf.utils.xyz_traj import read_xyzfile
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='xyz2json.py')
    parser.add_argument('XYZFILE', help='xyz file')
    args = parser.parse_args()
    
    traj = read_xyzfile(args.XYZFILE)
    print(json.dumps(traj, indent=2))