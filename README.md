# Master Thesis
This repository contains all the code generated for my master thesis. It contains multiple files. Below is a short list explaining each file:
- PINN Code
    - ODE_SBC.py and ODE_HBC.py are the PINN codes for the first problem solved the simple smooth ODE.
    - ODE2_HBC.py is the PINN code for the second problem solved, the ODE with the discontinuity.
    - PINN_Phasefield_Strong.py, PINN_Phasefield_Weak.py are the PINN codes for the phase field problems. Once in its strong form and its weak. (Only the weak one gives correct predictions).
- KANN Code
    - kann_variants.py + parameters.py, are the KANN codes for the first problem.
    - kann_ode_hde.py + parameters.ode.py, are the KANN codes for the second problem.
    - kann_ode_hde_param.py + parameters_ode_param.py, are the KANN codes for the parametrized problem.
    - kann_ode_1D_Multi.py + parameters_ode_multidim.py, are testing files to see how it behaves for a multidimensional output.
    - kann_ode_multidim.py + parameters_ode_multidim.py, are testing files to see how it behaves for a multidimensional output.
    - kann_phasefield_strong.py and kann_phasefield_weak.py + parameters_phasefield_strong.py and parameters_phasefield_weak.py are the KANN files for the phase field problem solved using either the strong or weak formulation. (Only the weak one gives correct predictions)

# Requirements
The requerements can be found in:
```
requirements.txt
```
and can be installed via pip:
```
pip install -r requirements.pip
```
# Bugs
The speedup of using cached values for the inner basis function only works for manual differentiation, as pytorch uses dynamical computational graph creation. This is for all the different problem cases.
