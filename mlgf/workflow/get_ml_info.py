import os
import numpy as np
from pyscf import scf, lib, cc
import joblib

import argparse
import json
import pandas as pd

from fcdmft.ac.pade import PadeAC
from fcdmft.ac.two_pole import TwoPoleAC


from mlgf.workflow.model_errors import validations_to_table
from mlgf.lib.ml_helper import get_sigma_ml, get_vmo, sigma_lo_mo, get_pade18
from mlgf.lib.dm_helper import get_dm_dyson, get_dm_linear, make_frozen_no
from mlgf.lib.dos_helper import calc_dos

from scipy.optimize import newton

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def get_quasiparticle_energies(acobj, mf_mo_energy, vk_minus_vxc = None):
    if vk_minus_vxc is None:
        vk_minus_vxc = np.zeros((len(mf_mo_energy), len(mf_mo_energy)))
    # self-consistently solve QP equation
    norbs = len(mf_mo_energy)
    qp_energy = np.zeros(norbs)
    for p in range(norbs):
        ac_orb_p = acobj[p, p]
        def quasiparticle(omega):
            sigmaR = ac_orb_p.ac_eval(omega).real
            return omega - mf_mo_energy[p] - (sigmaR.real + vk_minus_vxc[p, p])
        try:

            e = newton(quasiparticle, mf_mo_energy[p], tol=1e-6, maxiter=100)
            qp_energy[p] = e
        except RuntimeError:
            qp_energy[p] = np.nan

    return qp_energy

def get_z(sigma_diag12, omega_fit12):
    dsig = (sigma_diag12[:,1].imag - sigma_diag12[:,0].imag)/(omega_fit12[1].imag-omega_fit12[0].imag)
    return 1./(1-dsig)

def get_properties(
    sigma,
    mlf,
    freqs,
    eta,
    properties="dq",
    nroot=1,
    ac_method="pade",
    linearized_dm=True,
    bse_active_space=None,
    ac_idx=None,
    diag_dos = True
):
    """Get properties from GW self-energy and DFT calc, self-energy can be from ML or true

    Args:
        sigma (np.complex64): self-energy on iomega points
        mlf (Data): Data.load_chk(mlf_chkfile)
        freqs (np.float64): real freqs to compute DOS
        eta (float): DOS band-broadening
        properties (str, optional): characters for properties to compute (d = dos, q = qpe, = bse, m = density matrix). Defaults to 'dq'.
        nroot (int, optional): bse nroot. Defaults to 20.
        ac_method (str, optional): AC method for getting spectrum and qpe from sigma(iomega). Defaults to 'pade'.
        linearized_dm (bool, optional): wether to do linearized dyson for density matrix integration step. Defaults to True, the convention for G0W0.
        bse_active_space (list, optional): see get_bse_singlets. Defaults to None.
        ac_idx (list, optional): indicies of iomega to use for AC for qpe and dos. Defaults to None.

    Returns:
        dict: dictionary of all the computed properties
    """    
    mlf_chkfile = getattr(mlf, 'fname', None)
    xc = getattr(mlf, 'xc', 'hf')
    if type(xc) is bytes:
        xc = xc.decode('utf-8')
    mf_mo_energy = mlf['mo_energy']
    omega_fit = getattr(mlf, 'omega_fit', None)

    nmo = len(mf_mo_energy)
    nocc = mlf['nocc']

    if xc == 'hf': 
        vk_minus_vxc = np.zeros((len(mf_mo_energy), len(mf_mo_energy)))
    else: 
        vk, v_mf = get_vmo(mlf)
        vk_minus_vxc = vk - v_mf

    if omega_fit is None:
        omega_fit, _ = get_pade18()
        omega_fit = mlf['ef'] + 1j*omega_fit

    if ac_idx is None:
        ac_idx = np.arange(sigma.shape[-1])

    # analytic continuation
    if ac_method == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), nocc)
    elif ac_method == 'pade':
        acobj = PadeAC()
        acobj.idx = ac_idx
    elif ac_method == 'pes':
        raise NotImplementedError
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(ac_method)))

    acobj.ac_fit(sigma, omega_fit, axis=-1)

    outputs = {}
    if 'q' in properties:
        qpe = get_quasiparticle_energies(acobj, mf_mo_energy, vk_minus_vxc = vk_minus_vxc)
        outputs['qpe'] = qpe
    
    if 'd' in properties:
        dos = calc_dos(freqs, eta, acobj, mf_mo_energy, vk_minus_vxc = vk_minus_vxc, diag = diag_dos) # diag 
        outputs['dos'] = dos

    if 'm' in properties:
        if linearized_dm:
            outputs['dm'] = get_dm_linear(sigma, mlf['omega_fit'], mlf['wts'], mf_mo_energy, vk_minus_vxc)
        else:
            outputs['dm'] = get_dm_dyson(sigma, mlf['omega_fit'], mlf['wts'], mf_mo_energy, vk_minus_vxc)
    
    if 'b' in properties and 'q' in properties:
        # raise NotImplementedError("On the fly BSE calculation not available. Use get_bse.py instead.")
        from mlgf.workflow.get_bse import get_bse_singlets
        exci_s, dipole, oscillator_strength, bse_active_space = get_bse_singlets(mlf_chkfile, qpe, nmo, nocc, nroot = nroot, bse_active_space = bse_active_space, xc = xc)
        outputs['bse_exci_s'] = exci_s
        outputs['bse_dipole'] = dipole
        outputs['bse_oscillator_strength'] = oscillator_strength
        outputs['bse_active_space'] = bse_active_space

    return outputs

def get_properties_cc(
    sigma,
    mlf,
    freqs,
    eta,
    properties="d",
    frozen=None,
    no_thresh=1e-4,
    ac_idx=None,
    z_idx=[0, 3],
    do_comparisons=False,
    ac_method="pade",
):
    """Get properties from CC self-energy and DFT calc, self-energy can be from ML or true

    Args:
        sigma (np.complex64): self-energy on iomega points
        mlf (Data): Data.load_chk(mlf_chkfile)
        freqs (np.float64): real freqs to compute DOS
        eta (float): DOS band-broadening
        properties (str, optional): characters for properties to compute (d = dos, e = FNO energies, m = density matrix). Defaults to 'd'.
        nroot (int, optional): bse nroot. Defaults to 20.
        ac_idx (list, optional): indicies of iomega to use for AC for qpe and dos. Defaults to None.
        z_idx (list, optional): indicies of iomega for FD gradient in z
        frozen (list, optional): indicies of FNO to use for FNO-CCSD, if supplied no_thresh ignored
        freqs (float): NO occupation threshold for constructing FNO space        
        do_comparisons (boolean): do all comparisons to various CC energy routines if e in properties
    Returns:
        dict: dictionary of all the computed properties
    """    
    xc = getattr(mlf, 'xc', 'hf')
    if type(xc) is bytes:
        xc = xc.decode('utf-8')
    mf_mo_energy = mlf['mo_energy']
    omega_fit = mlf['omega_fit']
    nmo, nocc = len(mf_mo_energy), mlf['nocc']

    if omega_fit is None:
        omega_fit, _ = get_pade18()
        omega_fit = mlf['ef'] + 1j*omega_fit

    if ac_idx is None:
        ac_idx = np.arange(sigma.shape[-1])

    # analytic continuation
    if ac_method == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), nocc)
    elif ac_method == 'pade':
        acobj = PadeAC()
        acobj.idx = ac_idx
    elif ac_method == 'pes':
        raise NotImplementedError
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(ac_method)))

    acobj.ac_fit(sigma, omega_fit, axis=-1)

    if xc == 'hf': 
        vk_minus_vxc = np.zeros((len(mf_mo_energy), len(mf_mo_energy)))
    else: 
        vk, v_mf = get_vmo(mlf)
        vk_minus_vxc = vk - v_mf

    outputs = {}
    outputs['no_thresh'] = no_thresh
    if 'd' in properties:
        dos = calc_dos(freqs, eta, acobj, mf_mo_energy, vk_minus_vxc=vk_minus_vxc, diag=False)
        outputs['dos'] = dos

        
    if 'm' in properties:

        outputs['dm'] = get_dm_dyson(sigma, omega_fit, mlf['wts'], mf_mo_energy, vk_minus_vxc)

        # frozen NOs from CCGF
        if 'e' in properties:
            if frozen is None:
                frozen, no_coeff = make_frozen_no(outputs['dm'], mf_mo_energy, mlf['mo_coeff'], mlf['nocc'], thresh=no_thresh)
                # print(frozen)
            else:
                frozen, no_coeff = make_frozen_no(outputs['dm'], mf_mo_energy, mlf['mo_coeff'], mlf['nocc'], nvir_act = nmo - nocc - len(frozen))
                # print(frozen)
            mol = lib.chkfile.load_mol(mlf.fname)
            mol.verbose = 0
            mf = scf.RHF(mol)
            # frozen = np.array([0, 1, 2] + list(frozen))
            mf.__dict__.update(lib.chkfile.load(mlf.fname, 'scf'))
            mycc = cc.CCSD(mf, frozen=frozen, mo_coeff=no_coeff)
            mycc.kernel()
            outputs['e_ccsd_ccgf'] = mycc.e_tot
            print(mlf.fname, frozen, mycc.e_tot)
            outputs['et_ccsdt_ccgf'] = mycc.ccsd_t()
            outputs['frozen_ccgf'] = frozen

        # frozen from cc.make_rdm1() in pyscf RCCSD
        if 'e' in properties and do_comparisons:
            frozen, _ = make_frozen_no(outputs['dm'], mf_mo_energy, mlf['mo_coeff'], mlf['nocc'], thresh=no_thresh)
            frozen, no_coeff = make_frozen_no(mlf['dm_cc'], mf_mo_energy, mlf['mo_coeff'], mlf['nocc'], nvir_act = nmo - nocc - len(frozen))
            print(frozen)
            mol = lib.chkfile.load_mol(mlf.fname)
            mol.verbose = 5
            mf = scf.RHF(mol)
            mf.__dict__.update(lib.chkfile.load(mlf.fname, 'scf'))
            mycc = cc.CCSD(mf, frozen=frozen, mo_coeff=no_coeff)
            mycc.kernel()
            outputs['e_ccsd_rdm1'] = mycc.e_tot
            outputs['et_ccsdt_rdm1'] = mycc.ccsd_t()
            outputs['frozen_rdm1'] = frozen


        # frozen MOs
        # TODO: fix this frozen MO part (though not used much in practice)
        # if 'e' in properties and do_comparisons:
        #     mol = lib.chkfile.load_mol(mlf.fname)
        #     mol.verbose = 5
        #     mf = scf.RHF(mol)
        #     # frozen = np.array([0, 1, 2] + list(frozen))
        #     mf.__dict__.update(lib.chkfile.load(mlf.fname, 'scf'))
        #     frozen = frozen[frozen_start:]
        #     mycc = cc.CCSD(mf, frozen=frozen)
        #     mycc.kernel()
        #     outputs['e_ccsd_mo_frozen'] = mycc.e_tot
        #     outputs['et_ccsdt_mo_frozen'] = mycc.ccsd_t()
        
        # release pyscf memory
        if 'e' in properties:
            del mol
            del mycc
            del mf

        if 'n' in properties:
            no_occ, no_coeff = np.linalg.eigh(outputs['dm'])
            outputs['no_occ'] = no_occ
            outputs['no_coeff'] = no_coeff

    if 'z' in properties:
        sigma_diag = np.diagonal(sigma[:,:,z_idx]).T
        outputs['z'] = get_z(sigma_diag, omega_fit[z_idx])
        # print(outputs['z'][12], outputs['z'][13])

    return outputs

def do_validation_gw(job):
    """wrapper around get_properties to compute GW properties for ML and True with MPI parallel

    Args:
        job (dict): dictionary of configuration variables for doing property calculations from self-energy
    """    

    model_file, mlf_chkfile, out_fname, validation_name = job['model_file'], job['mlf_chkfile'], job['output_file'], job['validation_name']
    basis_name, freqs, eta, properties, save_sigma = job['basis_name'], job['freqs'], job['eta'], job['properties'], job['save_sigma'] 
    
    linearized_dm = job['linearized_dm']
    bse_active_space, bse_nroot = job['bse_active_space'], job['bse_nroot']

    ac_idx = job['ac_idx']

    out_dict = {}
    model_obj = joblib.load(model_file)
    sigma_ml = model_obj.predict_full_sigma(mlf_chkfile)
    if save_sigma:
        out_dict[f'sigma_ml_{basis_name}'] = sigma_ml
        uncertainty = getattr(model_obj, "uncertainty_full_sigma", None)
        if uncertainty is not None:
            if callable(uncertainty):
                sigma_ml_uncertainty = model_obj.uncertainty_full_sigma(mlf_chkfile, remount_mlf_chkfile = False)
                out_dict['sigma_ml_uncertainty'] = sigma_ml_uncertainty

    mlf = model_obj.pdset[0]
    C_lo_mo = mlf[f'C_{basis_name}_mo']
    
    sigma_ml = sigma_lo_mo(sigma_ml, C_lo_mo) 

    out_dict_ml = get_properties(sigma_ml, mlf, freqs, eta, properties = properties, nroot = bse_nroot,linearized_dm = linearized_dm, bse_active_space = bse_active_space, ac_idx = ac_idx)
    
    for k in out_dict_ml.keys():
        out_dict[f'{k}_ml'] = out_dict_ml[k]

    try:
        sigma_true = mlf[f'sigma_{basis_name}']
        sigma_true = sigma_lo_mo(sigma_true, C_lo_mo)   
        out_dict_true = get_properties(sigma_true, mlf, freqs, eta, properties = properties, nroot = bse_nroot, linearized_dm = linearized_dm, bse_active_space = out_dict_ml.get('bse_active_space', None), ac_idx = ac_idx) # use same BSE active space as ML

        for k in out_dict_true.keys():
            out_dict[f'{k}_true'] = out_dict_true[k]

    except KeyError as e:
        print(f'KeyError for true values: {str(e)} on {mlf_chkfile}')
        if 'qpe' in mlf.keys():
            print(f'mlf_chkfile {mlf_chkfile} has qpe, assigning to true values instead of from sigmaI')
            out_dict['qpe_true'] = mlf['qpe']


    # out_dict['rsq_df'] = df
    out_dict['model_file'] = model_file
    out_dict['mlf_chkfile'] = mlf_chkfile
    out_dict['freqs'] = freqs
    out_dict['eta'] = eta
    out_dict['basis_name'] = basis_name
    out_dict['nocc'] = mlf['nocc']
    out_dict['mo_energy'] = mlf['mo_energy']
    out_dict['xc'] = getattr(mlf, 'xc', 'hf')#.decode('utf-8')
    out_dict['ef'] = mlf['ef']
    lib.chkfile.save(out_fname, validation_name, out_dict)


def do_validation_cc(job):
    """wrapper around get_properties to compute CC properties for ML and True with MPI parallel

    Args:
        job (dict): dictionary of configuration variables for doing property calculations from self-energy
    """    

    model_file, mlf_chkfile, out_fname, validation_name = job['model_file'], job['mlf_chkfile'], job['output_file'], job['validation_name']
    basis_name, freqs, eta, properties, save_sigma = job['basis_name'], job['freqs'], job['eta'], job['properties'], job['save_sigma'] 
    no_thresh = job.get('no_thresh', 1e-3)
    ac_idx = job.get('ac_idx', None)
    z_idx = job.get('z_idx', [0, 3])

    out_dict = {}
    model_obj = joblib.load(model_file)
    sigma_ml = model_obj.predict_full_sigma(mlf_chkfile)
    if save_sigma:
        out_dict[f'sigma_ml_{basis_name}'] = sigma_ml
        uncertainty = getattr(model_obj, "uncertainty_full_sigma", None)
        if uncertainty is not None:
            if callable(uncertainty):
                sigma_ml_uncertainty = model_obj.uncertainty_full_sigma(mlf_chkfile, remount_mlf_chkfile = False)
                out_dict['sigma_ml_uncertainty'] = sigma_ml_uncertainty

    mlf = model_obj.pdset[0]
    C_lo_mo = mlf[f'C_{basis_name}_mo']

    sigma_ml = sigma_lo_mo(sigma_ml, C_lo_mo) 

    out_dict_ml = get_properties_cc(sigma_ml, mlf, freqs, eta, properties = properties, no_thresh = no_thresh, ac_idx = ac_idx, z_idx = z_idx, do_comparisons = False)
    
    for k in out_dict_ml.keys():
        out_dict[f'{k}_ml'] = out_dict_ml[k]

    try:
        sigma_true = mlf[f'sigma_{basis_name}']
        sigma_true = sigma_lo_mo(sigma_true, C_lo_mo)
        out_dict_true = get_properties_cc(sigma_true, mlf, freqs, eta, properties = properties, frozen = out_dict_ml.get('frozen_ccgf', None), ac_idx = ac_idx, z_idx = z_idx) # use same BSE active space as ML

        for k in out_dict_true.keys():
            out_dict[f'{k}_true'] = out_dict_true[k]

    except KeyError as e:
        print(f'KeyError for true values: {str(e)} on {mlf_chkfile}')
        if 'qpe' in mlf.keys():
            print(f'mlf_chkfile {mlf_chkfile} has qpe, assigning to true values instead of from sigmaI')
            out_dict['qpe_true'] = mlf['qpe']


    # out_dict['rsq_df'] = df
    out_dict['model_file'] = model_file
    out_dict['mlf_chkfile'] = mlf_chkfile
    out_dict['freqs'] = freqs
    out_dict['eta'] = eta
    out_dict['basis_name'] = basis_name
    out_dict['nocc'] = mlf['nocc']
    out_dict['mo_energy'] = mlf['mo_energy']
    out_dict['xc'] = getattr(mlf, 'xc', 'hf')#.decode('utf-8')
    out_dict['ef'] = mlf['ef']
    lib.chkfile.save(out_fname, validation_name, out_dict)
    del model_obj
    

def predict_sigma(model_file, mlf_chkfile, return_uncertainty = False):
    """loads the pcikled model object from joblib and uses it to get a self-energy calculation for the DFT calculation stored in mlf_chkfile

    Args:
        model_file (string): pickled file for self-energy predictor
        mlf_chkfile (string): chkfile with DFT calculation and maybe GW calculation
        return_uncertainty (boolean, optional): get the self-energy and the uncertainty. Defaults to False
    """    

    sigma_ml = get_sigma_ml(mlf_chkfile, model_file)
    
    if return_uncertainty: 
        model = joblib.load(model_file)
        sigma_ml_uncertainty = model.uncertainty_full_sigma(mlf_chkfile, remount_mlf_chkfile = True)
        return sigma_ml, sigma_ml_uncertainty
    else:
        return sigma_ml

def make_outputs(jobs, defaults):
    """Yet another wrapper around property prediction for MPI

    Args:
        jobs (list): list of dictionaries containing all variables needed for each ML validation
        defaults (dict): default config variables (e.g. kwargs for get_properties())
    """    
    defaults['validation_name'] = 'validation'
    for job in jobs:
        for key in defaults.keys():
            job[key] = job.get(key, defaults[key])

        pts = job.get('pts', defaults['pts'])
        job['freqs'] = np.linspace(-1, 1, pts)
        if job['gf'] == 'gw':
            do_validation_gw(job)
        if job['gf'] == 'cc':
            do_validation_cc(job)

def make_outputs_mpi(jobs, defaults):
    indices = np.arange(len(jobs))
    indices_subset = indices[indices % size == rank]
    # jobs_rank = [jobs[i] for i in range(len(jobs)) if i % size == rank]
    make_outputs(jobs[indices_subset], defaults)
    
def generate_validation_queue(model_files, validation_dir, output_dir):
    validation_data = []
    validation_files = os.listdir(validation_dir)
    indices = np.arange(len(validation_files))
    indices_subset = indices[indices % size == rank]
    validation_files_subset = [validation_files[i] for i in indices_subset]

    for model_file in model_files:
        for validation_file in validation_files_subset:
            
            # Create output file path
            output_file = os.path.join(output_dir, validation_file)
            
            # Create dictionary
            data = {
                'model_file': model_file,
                'mlf_chkfile': f'{validation_dir}/{validation_file}',
                'output_file': output_file,
                'validation_name': model_file
            }
            
            # Append dictionary to the list
            validation_data.append(data)

    return validation_data

def get_file_subset_by_rank(output_dir, rank, size):
    validation_files = os.listdir(output_dir)
    indices = np.arange(len(validation_files))
    indices_subset = indices[indices % size == rank]
    validation_files_subset = [f'{output_dir}/{validation_files[i]}' for i in indices_subset]
    print(f'For src: {output_dir}, Rank: {rank} has {len(validation_files_subset)} validation files to proesses.')
    return validation_files_subset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='get_ml_info.py')
    # must have key jobs with a list of dictionaries, each with key model_file, output_file, mlf_chkfile
    parser.add_argument('--json_spec', required=True, help='json file with all the property prediction metadata')
    # by default True
    parser.add_argument('--no_new_folders', action = 'store_true', help='Do not go through all the jobs and make output folders from eacb output_file if they do not exist')
    parser.add_argument('--output_csv', type=str, required = False, help="csv to write outputs to, no csv created if not supplied")

    args = parser.parse_args()
    json_spec = args.json_spec
    no_new_folders = args.no_new_folders

    defaults = {'pts' : 201, 'eta' : 0.01, 'basis_name' : 'saiao', 'properties' : 'dq', 'zero_coulomb' : False, 'zero_core' : False, 'save_sigma' : False, 'linearized_dm' : True, 'bse_nroot' : 10, 'gf' : 'gw', 'no_thresh' : 1e-3, 'ac_idx' : None, 'z_idx' : [0, 3]}

    assert('.json' in json_spec)
    with open(json_spec) as f:
        spec = json.load(f)
    if 'jobs' in spec.keys():
        jobs =  spec['jobs']

        if rank == 0 and not no_new_folders:
            for job in jobs:
                outdir = '/'.join(job['output_file'].split('/')[:-1])
                
                if not os.path.exists(outdir):
                    print(f'Making dir {outdir}')
                    os.makedirs(outdir)
                elif not os.path.isdir(outdir):
                    raise FileExistsError(f'{outdir} exists but is not a directory')
                    comm.Abort()

        comm.Barrier()
        make_outputs_mpi(jobs, defaults)

        comm.Barrier() # unnecessary but can't hurt
    else:
        output_dir = spec['validation_dir']
        input_dir = spec['input_dir']
        defaults['linearized_dm'] = spec.get('linearized_dm', defaults['linearized_dm'])
        defaults['properties'] = spec.get('properties', defaults['properties'])
        defaults['bse_active_space'] = spec.get('bse_active_space', None)
        defaults['bse_nroot'] = spec.get('bse_nroot', None)
        defaults['gf'] = spec.get('gf', defaults['gf'])
        defaults['no_thresh'] = spec.get('no_thresh', defaults['no_thresh'])
        defaults['ac_idx'] = spec.get('ac_idx', defaults['ac_idx'])
        defaults['z_idx'] = spec.get('z_idx', defaults['z_idx'])
        defaults['basis_name'] = spec.get('basis_name', defaults['basis_name'])
        
        comm.Barrier()
        if type(input_dir) == list:
            assert(type(output_dir) == list)
            assert(len(output_dir) == len(input_dir))
            jobs = []
            for i in range(len(output_dir)):
                new_folder = output_dir[i]

                if not os.path.exists(new_folder) and not no_new_folders and rank == 0:
                    print(f'Making dir {new_folder}')
                    os.makedirs(new_folder)
                new_jobs = generate_validation_queue(spec['model_files'], input_dir[i], new_folder)
                jobs = jobs + new_jobs
                
        else:
            if not os.path.exists(output_dir) and not no_new_folders and rank == 0:
                print(f'Making dir {output_dir}')
                os.makedirs(output_dir)
            jobs = generate_validation_queue(spec['model_files'], input_dir, output_dir)
        comm.Barrier()

        print(f'rank {rank} has {len(jobs)} jobs')
        make_outputs(jobs, defaults)
        comm.Barrier()

        if not args.output_csv is None:
            do_dm = 'm' in defaults['properties']
            if type(output_dir) == list:
                df_list = []
                for i in range(len(output_dir)):
                    validation_files_subset = get_file_subset_by_rank(output_dir[i], rank, size)
                    df_new = validations_to_table(validation_files_subset, spec['model_files'], do_dm = do_dm)
                    df_list.append(df_new.copy())
                df = pd.concat(df_list)
                
            else: 
                validation_files_subset = get_file_subset_by_rank(output_dir, rank, size) 
                df = validations_to_table(validation_files_subset, spec['model_files'], do_dm = do_dm)
            print(f'Rank {rank} finished collection into dataframe!')
            comm.Barrier()
        
            gathered_dfs = comm.gather(df)
            if rank == 0:
                gathered_dfs = pd.concat(gathered_dfs)
                gathered_dfs.to_csv(args.output_csv, index = False)
            comm.Barrier() 

    MPI.Finalize()



