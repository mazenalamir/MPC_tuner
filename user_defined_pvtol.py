'''
---------------------------------------------------------------
Author: Mazen Alamir
Institution: CNRS, Unicersity of Grenoble-Alpes
Date: August 2023
----
This file defines the use specific file for use in the myMPC
Package that optimize the NMPC parameters for successful real
time implementation via a given computational capacity.
---------------------------------------------------------------
'''

# -------------------------------------------
# NOTICE THAT ALL THE INVOLVED FUNCTIONS
# SHOULD BE DEFINED USING VERTCAT AND
# CASADI COMPATIBLE EXPRESSIONS!
# (except for the generate_cloud map)
# -------------------------------------------

import numpy as np
from casadi import vertcat

class Container:

    # Generic structure for report

    def __init__(self):
        pass

# define the dimensions
pvtol = Container()
pvtol.nx, pvtol.nu = 6, 2    # Dimension of the state and control
pvtol.n_p = 2        # number of parameters in the model
pvtol.n_q = 4         # number of parameter needed in the cost function

# Define the model parameters
pvtol.p = np.array([0.04, 1.0])

# Define the cost param
pvtol.q = vertcat(1.2, -0.8, 1.0, np.pi)

# Define the saturation on the control
pvtol.u_min = np.array([-5e1, -5e1])
pvtol.u_max = np.array([+5e1, +5e1])

# Define the bounds on x and q
pvtol.x_min = np.array([-2, -2, -0.8 * np.pi, -0.1, -0.1, -0.1])
pvtol.x_max = -pvtol.x_min
pvtol.q_min = [-1, -1, 1, np.pi]
pvtol.q_max = [+1, +1, 1, np.pi]

# Define the nominal values and the std on the parameters
pvtol.p_nom = pvtol.p
pvtol.p_std = 0.1

# Define the sampling period and the prediction horizon
# the sampling period used in the MPC can be different
# here tau is the smallest one compatible with the integration
# of the ODE using a 4th-order Runge Kutta scheme.
pvtol.tau = 0.02

# number of point-wise constraints
pvtol.nc = 4

#------
# Define the ODE  function
def p_vtol(x, u, p):

    epsilon, gain = p[0], p[1]
    # Note the systematic use of vertcat for Casadi compatibility
    xdot = vertcat(
        x[3], x[4], x[5],
        -u[0] * np.sin(x[2]) + epsilon * u[1] * np.cos(x[2]),
        u[0] * np.cos(x[2]) + epsilon * u[1] * np.sin(x[2]) - 1,
        gain * u[1]
    )
    return xdot
pvtol.ode = p_vtol
#------
# Define the constraints function
def constraints(x, u, p, q):

    # For a rational choice of the constraints violation
    # threshold, its is strongly recommended to define the
    # constraints in a normalized way. so that all the components
    # of the constraints compare to unity.
    # as an example, below we use (x[5]/d_theta_dt_max-1<=0) rather
    # than (x[5]-d_theta_dt_max<=0)

    d_theta_dt_max = q[2]
    theta_max = q[3]
    # Note the systematic use of vertcat for Casadi compatibility
    c = vertcat(
        x[5]/d_theta_dt_max-1,
        -x[5]/d_theta_dt_max-1,
        x[2]/theta_max - 1,
        -x[2]/theta_max - 1
    )
    return c
pvtol.constraints = constraints
#------
# Define the stage cost
def stage_cost(x, u, p, q):

    # This is the pure stage cost not including the
    # penalty on the constraints' violation.

    qx = [1e3, 1e3, 1e3, 1, 1, 1]
    qu = [1e-1, 1e-1]
    #---
    ud = [1, 0]
    yd, zd = q[0], q[1]
    xd = vertcat(yd, zd, 0, 0, 0, 0)
    #---
    ell = 0
    for i in range(6):
        ell += qx[i] * (x[i]-xd[i]) ** 2
    for i in range(2):
        ell += qu[i] * (u[i]-ud[i]) ** 2
    return ell
pvtol.stage_cost = stage_cost
#------
# Define the final penalty function
def final_penalty(x, p, q):

    yd = q[0]
    zd = q[1]
    # Note the systematic use of vertcat for Casadi compatibility
    xd = vertcat(yd, zd, 0, 0, 0, 0)
    q_xf = np.array([1e3, 1e3, 1e3, 1, 1, 1])
    ef = 0
    for i in range(6):
        ef += q_xf[i] * (x[i] - xd[i]) ** 2
    return ef
pvtol.final_penalty = final_penalty
#------
# Generate a cloud of scenarios, each defined by a triplet
# (x0, p, q)
#-----
def generate_cloud(nSamples=None):

    # Generate nSamples scenarios with different initial state, model's parameters
    # and set-points (cost parameters)

    x_min, x_max = pvtol.x_min, pvtol.x_max
    q_min, q_max = pvtol.q_min, pvtol.q_max
    p_nom, p_std = pvtol.p_nom, pvtol.p_std

    dX = np.array(x_max)-np.array(x_min)
    X = [np.array(x_min) + np.random.rand(len(x_min)) * dX
                                for _ in range(nSamples)]

    P = [np.array(p_nom) * (1 + p_std * np.random.randn(len(p_nom)))
         for _ in range(nSamples)]

    dQ = np.array(q_max) - np.array(q_min)
    Q = [np.array(q_min) + np.random.rand(len(q_min)) * dQ
         for _ in range(nSamples)]

    A_sc = Container()
    A_sc.x0, A_sc.p, A_sc.q = X, P, Q

    return A_sc

pvtol.generate_cloud = generate_cloud