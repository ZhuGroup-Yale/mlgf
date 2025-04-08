# mlgf
Machine learning many-body Green's function and self-energy

Authors: Christian Venturella, Jiachen Li, Christopher Hillenbrand, Tianyu Zhu


Features
--------

- Many-body Green's function calculations with fcmdft: $GW$ theory, coupled cluster theory, and FCI 
- Predicting the many-body Green's function (MBGF) by targeting the self-energy from DFT features
- Local and equivariant represenations of electronic structure data with libdmet 
- Various machine learning algorithms for fitting a self-energy functional (KRR and GNN)
- Chemical properties dervied from the MBGF
  - photoemission spectrum (i.e. density of states)
  - $GW$ quasiparticle energies
  - quasiparticle renormalization
  - 1-particle density matrix and downstream observables:
    - dipoles
    - quadrupoles
    - IAO partial charges
    - FNO-CCSD energy
  - optical spectrum with $GW$-BSE.

Installation
------------

### Requirements
  * PySCF and all dependencies 
  * fcdmft (by Tianyu Zhu, https://github.com/ZhuGroup-Yale/fcdmft)
  * libdmet (by Zhi-Hao Cui, https://github.com/gkclab/libdmet_preview)
  * scikit-learn for self-energy KRR (https://scikit-learn.org)
  * PyTorch Geometric and all dependencies for MBGF-Net (https://pytorch-geometric.readthedocs.io, https://pytorch.org)
  * pandas for data analysis (https://pandas.pydata.org)

### Optional
  * CP2K, for ab initio molecular dynamics (`mlgf/aimd`)

### Method 1. Set PYTHONPATH
Set environment variable `PYTHONPATH`so python intepreter can find `mlgf`. 
For example, if mlgf is installed in `/opt`, you can run

```
export PYTHONPATH=/opt/mlgf:$PYTHONPATH
```

### Method 2. Pip installation
If you want to install different versions of `mlgf` in separate Python environments, or if you don't like environment variables, you can do an editable installation with Pip. This method also works for most Python packages in the wild. Clone the repo and run
```
pip install -e .
```
No dependencies will be installed in this step---you have to install them separately. If you don't intend to edit the code, and just want to run it, then remove `-e`
```
pip install .
```
Method 2 is recommended for new users, as it is [standard practice](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html). Consult [the pip documentation](https://pip.pypa.io/en/stable/cli/pip_install/) for further information about command-line options for pip.

References
----------

Cite the following papers for the MLGF workflow, KRR implementation, and PyG data processing and GNN architecture:

* C. Venturella, C. Hillenbrand, J. Li, and T. Zhu, Machine Learning Many-Body Green’s Functions for Molecular Excitation Spectra, J. Chem. Theory Comput. 2024, 20, 1, 143–154

* C. Venturella, J. Li, C. Hillenbrand, X. L. Peralta, J. Liu, T. Zhu, Unified Deep Learning Framework for Many-Body Quantum Chemistry via Green’s Functions; 2024. arXiv:2407.20384

Please cite the following papers in publications utilizing the fcdmft package for MBGF calculation:

* T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 17, 727-741 (2021)

* T. Zhu and G. K.-L. Chan, Phys. Rev. X 11, 021006 (2021)

