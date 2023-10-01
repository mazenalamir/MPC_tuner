# MPC_tuner
A python package for the optimization of NMPC implementation options

Nonlinear Model Predictive Control **(NMPC)** is the most advanced control design. It enables to take into account nonlinear dynamics, non conventional control objective through the definition of a cost function and the presence of input (control) and state constraints.

While the theoretical assessment of NMPC are now clear and freely available solvers are widely used, it remains true that the final implementation of NMPC involves the choice of the following parameters:

- The prediction horizon's length
- The penalties on the soft constraints and on the terminal cost
- The control parameterization (number of d.o.f)
- The sampling period for the integration inside the solver
- The 
