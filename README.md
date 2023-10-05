# MPC_tuner
A python package for the optimization of NMPC implementation options

## Citation
A full description of the principles and a detailed illustrative example are given in the paper below:

> Mazen Alamir. A framework and a `python`-package for real-time NMPC parameter settings. [arXiv:2309.17238](https://arxiv.org/abs/2309.17238), October, 2023.

## Recall on Nonlinear Model Predictive Control

Nonlinear Model Predictive Control **(NMPC)** is the most advanced control design. It enables to take into account nonlinear dynamics, non conventional control objective through the definition of a cost function and the presence of input (control) and state constraints. It is generally based on the repetitive solution of optimal control problems of the following form (soft constraints are used except for the input control saturations):

$$
\min_{\mathbf u} J(\mathbf u\ \vert x_0,p):=\rho_f\Psi(x_N)+\sum_{k=1}^{N_\text{pred}} \ell(x_{k},u_{k-1})+ \rho_\text{cstr} \max_{i=1}^{n_c}\lfloor c_i(x_k,u_{k-1})\rfloor_+^2
$$

where 

- $\mathbf u:=(u_0,\dots,u_{N_\text{pred}-1})\in \mathbb [R^{n_u}]^{N_\text{pred}}$ is the sequence of control that minimizes the above cost function 
- $x_k$ are the next states starting from the initial state $x_0$ and given the dynamics: 

$$
\dot x = f(x,u,p)
$$

in which $p$ is a vector of model's parameters. Once this problem is solved the first action $u_0$ is applied to the system and the process is repeated in the next **updating instants**. 

The *raison d'être* of this package is that the above equation and principle still leave many undecided choices that might be mandatory to derive a real-time implementatable algorithm. The next section describes the implementation parameters that the package helps tuning. 

## The implementation parameters that are tuned by the MPC_tuner package

While the theoretical assessment of NMPC are now clear and freely available solvers are widely used, it remains true that the final implementation of NMPC involves the choice of the following parameters (see below for some illustrations):

- The prediction horizon's length
- The control updating period $\tau_u=\kappa \tau$ ($\tau$ is the largest sampling period making a Runge kutta integration scheme sufficiently consistent). 
- The penalties on the soft constraints $\rho_\text{cstr}$ and on the final cost $\rho_f$,
- The control parameterization (number of d.o.f) leading to a number `n_ctr` of decision instants over the prediction horizon,
- The sampling period $\tau_p=\dfrac{\tau_u}{\texttt{nstep}}$ for the integration inside the solver. More precisely the tuned parameter is $\mu_d\in [0,1]$ that appears in the following expression `n_steps` $=\lceil 1+\mu_d(\kappa-1)\rceil$
- The maximum number of iterations `iter_max` allowed for the solver at each updating compuation.

To summarize, the algorithm provided by the package enables to tune the following vector of implementation parameters: 

$$
\pi := \begin{bmatrix} \kappa\cr \mu_d\cr N_\text{pred}\cr n_\text{ctr}\cr \rho_f\cr \rho_\text{cstr}\cr \texttt{max-iter}\end{bmatrix}
$$

| Illustration   |      Parameter to tune     | 
|:----------:|:-------------:|
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_u.png" width="100%"> |  $\tau_u=\kappa\tau$ The control updating period | 
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/dof.png" width="100%"> |  The control horizon defining the number of decision instants `n_ctr` |
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_p.png" width="100%"> |  The sampling period for the integration inside the solver $\tau_p=\frac{\tau_u}{\lceil 1+\mu_d(\kappa-1)\rceil}$ | 

## The guiding principles 

The idea is to choose the implementation parameters such that some admissibility conditions are satisfied over **a set $\mathcal A$ of cerifying scenarios** that is randomly sampled. The cardinalty of the set is chosen based on the probabilistic certification rules. The admissibility conditions are described in the following section. 

### The admissibility conditions
The NMPC implementation parameters are chosen such that the following requirements are satisfied:

- The **real-time compatibility** meaning that the computation time is lower than the available time. The latter is computed on a targeted device that might be different from the one on which the package is used. This can be done by using the following threshold on the computation time: `dev_acc` $\times \tau_u$. More precisely, when the computation for a single compuation is lower than `dev_acc` $\times \tau_u$ and this for all the updating instants during the closed-loop simulation then real-time compatibility is conformed **for the simualted scenario**.
  
- **The contraction** of the cost function at the end of a predefined period of closed-loop simulation
  
- **The constraints satisfaction** at each instants over the closed-loop simulation. 

## Description of the package 

The figure below shows the main utilities provided by the package as well as the necessary user-defined elements that need to be provided: 

<div align="center">
<img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/packages.png" width="60%">
<p>Summary of the main classes and functions exported by the package and the user-defined object to prepare for the specific control problem.</p>
</div>

## Content of the repository

In the list of files, you can find a complete example illustrating the NMPC design for the control of a Planar Vertical Take Of and Landing (PVTOL) nonlinear system. 
More precisely, the repository contains two files:

- The python file `MPC_tuner.py` containing the package
- The python file `user_defined_pvtol.py` which contains the PVTOL related files to be used by the package utilities
- The jupyter notebook `test_MPC_tuner.ipynb` which uses the previous files to find the admissible MPC settings.

Note however that the call of the utitlity `Design_MPC` contained in the `test_MPC_tunner.ipynb uses small values of `N_trials`, `nb` and `nsb` for the sake of illustation. Results similar to the ones obtained in the paper might be obtained by using the corresponding values of the above parameters. 


