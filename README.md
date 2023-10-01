# MPC_tuner
A python package for the optimization of NMPC implementation options

Nonlinear Model Predictive Control **(NMPC)** is the most advanced control design. It enables to take into account nonlinear dynamics, non conventional control objective through the definition of a cost function and the presence of input (control) and state constraints. It is generally based on the repetitive solution of optimal control problems of the following form (soft constraints are used except for the input control saturations):

$$
\min_{\mathbf u} J(\mathbf u\ \vert x_0,p):=\rho_f\Psi(x_N)+\sum_{k=1}^{N} \ell(x_{k},u_{k-1})+ \rho_\text{cstr} \max_{i=1}^{n_c}\lfloor c_i(x_k,u_{k-1})\rfloor_+^2
$$

where $x_k$ are the next states starting from the initial state $x_0$ and given the dynamics: 

$$
\dot x = f(x,u,p)
$$

in which $p$ is a vector of model's parameters. 

While the theoretical assessment of NMPC are now clear and freely available solvers are widely used, it remains true that the final implementation of NMPC involves the choice of the following parameters (see below for some illustrations):

- The prediction horizon's length
- The control updating period $\tau_u=\kappa \tau$ ($\tau$ is the largest sampling period making a Runge kutta integration scheme sufficiently consistent). 
- The penalties on the soft constraints $\rho_\text{cstr}$ and on the final cost $\rho_f$,
- The control parameterization (number of d.o.f) leading to a number `n_ctr` of decision instants over the prediction horizon,
- The sampling period $\tau_p=\dfrac{\tau_u}{\texttt{nstep}}$ for the integration inside the solver. More precisely the tuned parameter is $\mu_d\in [0,1]$ that appears in the following expression `n_steps` $=\lceil 1+\mu_d(\kappa-1)\rceil$
- The maximum number of iterations `iter_max` allowed for the solver at each updating compuation.

| Illustration   |      Parameter to tune     | 
|:----------:|:-------------:|
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_u.png" width="100%"> |  $\tau_u=\kappa\tau$ The control updating period | 
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/dof.png" width="100%"> |  The control horizon defining the number of decision instants `n_ctr` |
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_p.png" width="100%"> |  The sampling period for the integration inside the solver $\tau_p=\frac{\tau_u}{\lceil 1+\mu_d(\kappa-1)\rceil}$ | 

To summarize, the algorithm provided by the package enables to tune the following vector of implementation parameters: 

$$
\pi :=
$$
