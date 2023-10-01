# MPC_tuner
A python package for the optimization of NMPC implementation options

Nonlinear Model Predictive Control **(NMPC)** is the most advanced control design. It enables to take into account nonlinear dynamics, non conventional control objective through the definition of a cost function and the presence of input (control) and state constraints.

While the theoretical assessment of NMPC are now clear and freely available solvers are widely used, it remains true that the final implementation of NMPC involves the choice of the following parameters:

- The prediction horizon's length
- The control updating period (Fig 1) 
- The penalties on the soft constraints and on the terminal cost
- The control parameterization (number of d.o.f) (Fig 2.) 
- The sampling period for the integration inside the solver (Fig 3.)

<p align="left">
  <img src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_u.png" width="40%">
  <p align="left" style="color:blue"> Fig. 1: The control updating period $\tau$ is the basic sampling period.</p>
  <img src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/dof.png" width="40%">
  <p align="left"> Fig. 2: The sampling period for integration inside the solver.</p>
  <img src="https://github.com/mazenalamir/MPC_tuner/blob/main/images/tau_p.png" width="40%">
  <p align="left"> Fig. 3: The sampling period for integration inside the solver.</p>
</p>

