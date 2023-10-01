# MPC_tuner
A python package for the optimization of NMPC implementation options

Nonlinear Model Predictive Control **(NMPC)** is the most advanced control design. It enables to take into account nonlinear dynamics, non conventional control objective through the definition of a cost function and the presence of input (control) and state constraints.

While the theoretical assessment of NMPC are now clear and freely available solvers are widely used, it remains true that the final implementation of NMPC involves the choice of the following parameters:

- The prediction horizon's length
- The control updating period $\tau_u=\kappa \tau$ ($\tau$ is the largest sampling period making a Runge kutta integration scheme sufficiently consistent). 
- The penalties on the soft constraints and on the terminal cost
- The control parameterization (number of d.o.f) (Fig 2.) 
- The sampling period $\tau_p$ for the integration inside the solver (Fig 3.)

| Illustration   |      Parameter to tune     | 
|:----------:|:-------------:|
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_u.png" width="100%"> |  $\tau_u=\kappa\tau$ The control updating period | 
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/dof.png" width="100%"> |  The control horizon defining the number of decision instants `n_ctr` |
|  <img align="center" src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_p.png" width="100%"> |  The sampling period for the integration inside the solver $\tau_p$ | 


<div align="center">
  
  <img src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_p.png" width="50%">
  <p align="center"> Fig. 3: The sampling period for integration inside the solver.</p>
</div>

