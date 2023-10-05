'''
---------------------------------------------------------------
Author: Mazen Alamir
Institution: CNRS, Unicersity of Grenoble-Alpes
Date: August 2023
----
This is the implementation of the framework proposed in the paper
entitled:

"A framework and a python-package for real-time NMPC
parameters setting"

The variables naming is inspired from the notation of the above
paper which make them sometimes not very self comprehensive.
Therefore, I strongly recommend the user to have a reading of the
paper in order to fully appreciate and use the present package.
---------------------------------------------------------------
'''

# 0) Make it that the constraint violation indicator represent a sort of percentage of
# violation so that the user can intuitively define a threshold of rejection.

# 0.5) Introduce a threshold on the constraint violation to stop the iteration in case of violation.
# This threshold should be a parameter of the optimization utility

# 0.7) use this threshold to add another stopping condition so that a setting (sigma, alpha) that violate
# the constraint break the loop on the set of scenarios to test.



import copy
import sys
import os
import numpy as np
import pandas as pd
from casadi import Function, MX, jacobian, vertcat, if_else, nlpsol
import matplotlib.pyplot as plt
from time import time as tt
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def silence_casadi_output():

    # This function is used to avoid the
    # extremely long casadi log that is not necessary here

    sys.stdout = open(os.devnull, 'w')

class Scenario:

    '''
    This class defines the ingredient of a simulation.
    '''

    def __init__(self, x0, p, q):

        # x0: The initial state for the scenario
        # p: The model's vector of parameters
        # q: The vector of parameters needed to define the optimal control problem
        #    (set-points, constraints bounds, etc.)

        self.x0 = x0
        self.p = p
        self.q = q

class Container:

    # Generic class for object with fields
    # (could have been replaced by the named_tuples object).

    def __init__(self):
        pass

class OptimPar:

    def __init__(self, gam=0.98, dev_acc=1.0, T=2.0, c_max=0.1, eps=0.05):

        self.gam = gam
        self.dev_acc = dev_acc
        self.T = T
        self.c_max = c_max
        self.eps = eps

class Sigma:
    '''
    An instance of this class contains the information regarding the
    sigma shaping vector of parameter (see the paper) that
    totally define the way the components of the design vector pi
    depend on the parameter alpha \in (0,1)

    N_pred_min: minimum value of the prediction horizon
    N_pred_max: maximum value of the prediction horizon

    power_max: max value of the power of parameters-shaping map phi
                (sigma \in (-power_max,...,-1,+1,...,+power_max)

    kappa_min: minimum values of tau_u/tau
    kappa_max: maximum values of tau_u/tau

    rho_constr_min: minimum value of the constraints weighting
    rho_constr_max: maximum value of the constraints weighting

    rho_final_min: minimum value of the terminal penalty
    rho_final_max: maximum value of the terminal penalty

    max_iter_min: minimum value of the max_iteration parameter
    max_iter_max: maximum value of the max_iteration parameter
    '''
    def __init__(self,
                 N_pred_min=5,
                 N_pred_max=25,
                 kappa_min=1,
                 kappa_max=10,
                 rho_constr_min=1e3,
                 rho_constr_max=1e7,
                 n_ctr_min = 3,
                 rho_final_min = 1e0,
                 rho_final_max=1e3,
                 sigma_max=10,
                 max_iter_min=5,
                 max_iter_max=20):

        # Randomly sample the  parameter shaping maps powers
        # to create the parameters' shaping dictionary of powers
        # This enables to define a scalar parameter (alpha) to be optimized
        # that scales between very fast to very low computational
        # burden. Each sampling of the power dictionary leads to
        # a different shaping of the relationship between the design
        # vector and the complexity parameter alpha.

        # Randomly sample the vector of powers sigma
        powers = [i+1 for i in range(sigma_max)] + [-(i+1) for i in range(sigma_max)]
        power_dict = {

            'N': random.choice(powers),
            'kappa': random.choice(powers),
            'n_steps':random.choice(powers),
            'rho_constr': random.choice(powers),
            'rho_final': random.choice(powers),
            'I_dec': random.choice(powers),
            'max_iter': random.choice(powers)
        }

        def phi(alpha,pow):
            '''
            The phi-function defined in the paper in order to shape the relationship
            between the value of alpha and the value of the component of the design
            parameter vector pi (see the paper)
            '''
            if pow>0:
                y = alpha ** pow
            else:
                y = alpha ** (1 / abs(pow))
            return y

        # Computing the NMPC alpha-dependent parameters maps pi_i(alpha) using the power dictionary sampled above.
        # Notice how the maps are defined such that higher alpha induce longer computations but better ideal results
        # should the real-time concern be neglected.


        self.N_pred = lambda alpha: int(N_pred_min + phi(alpha, power_dict['N']) * (N_pred_max - N_pred_min))
        self.kappa = lambda alpha: int(kappa_max - phi(alpha, power_dict['kappa']) * (kappa_max - kappa_min))
        # Notice here that the parameters n_steps is directly used rather than mu_d in the paper.
        self.n_steps = lambda alpha: int(np.ceil(1 + phi(alpha, power_dict['n_steps']) * (self.kappa(alpha) - 1)))
        self.rho_constr = lambda alpha: rho_constr_min + phi(alpha, power_dict['rho_constr']) * (rho_constr_max - rho_constr_min)
        self.rho_final = lambda alpha: rho_final_min + phi(alpha, power_dict['rho_final']) * (rho_final_max - rho_final_min)
        self.I_dec = lambda alpha: np.arange(1, n_ctr_min + int(phi(alpha, power_dict['I_dec']) * self.N_pred(alpha)))
        self.max_iter = lambda alpha: int(max_iter_min + phi(alpha, power_dict['max_iter']) * (max_iter_max - max_iter_min))

        self.show_df(ngrid=100)

    def show_df(self, ngrid=20):
        '''
        Dataframe form of the maps pi_i(alpha)
        This dataframe is also used to plot these maps.
        '''
        alphas = np.logspace(-2, 0, ngrid)
        df = []
        for i in range(ngrid):

            alpha = alphas[i]
            N_pred = self.N_pred(alpha)
            kappa = self.kappa(alpha)
            n_steps = self.n_steps(alpha)
            rho_constr = self.rho_constr(alpha)
            rho_final = self.rho_final(alpha)
            n_dec = len(self.I_dec(alpha))
            max_iter = self.max_iter(alpha)
            cols = ['Prediction Horizon Length', 'kappa', 'n_steps',
                    'Constraints Penalty', 'Final Penalty', 'Number decision instants', 'max_iter']
            dfi = pd.DataFrame([N_pred, kappa, n_steps, rho_constr, rho_final, n_dec, max_iter], index=cols).T
            df += [dfi]

        df = pd.concat(df, axis=0)
        df.index = alphas
        self.df = df

        return df

    def plot_phi(self, ngrid=100, figsize=(16, 20)):

        # Plots the shaping maps pi_i(alpha) for the instantiated sigma.

        df = self.show_df(ngrid=ngrid)
        ix = 0
        mus = np.array(df.index)

        fig = plt.figure(figsize=figsize)

        ax = []
        for c in df.columns:
            ax.append(fig.add_subplot(3, 3, ix + 1));
            ax[ix].plot(mus, df[c].values, lw=3);
            ax[ix].grid(True);
            ax[ix].set_title(c, size=18);
            ax[ix].set_xlim([mus[0], mus[-1]]);
            ax[ix].set_xlabel(r'$\alpha$', size=18)
            ax[ix].tick_params(axis='both', labelsize=12)
            ix += 1
        return fig

class MPC:
    '''
    The class that defined the MPC problem

    the instantiation of this depends on container pb with the following field

    ode(x, u, p): The ordinary differential equation governing the system
    constraints(x, u, p, q) : the stage constraints other than control saturation
    nx, nu: dimension of the state and the control input vectors respectively
    n_p: number of model's parameters
    n_q: number of parameters involved in the definition of the optimal control problem
    tau : the sampling period for the integration of the ODEs.
    N_prediction: the prediction horizon
    u_min, u_max: vector bounds on the control input
    nc: the number of constraints at each instant of the prediction horizon
    stage_cost(x, u, p, q): the stage cost
    final_penalty(x, p, q): the terminal penalty on the state
    I_decision: list of instants where control is free (other than 0)
    rho_constr: penalty on the constraints' violation.
    max_iter: maximum_number of iteration for the solver
    kappa: the ratio between the integration step and the MPC step.

    NOTA:

    use constraints=None if no constraints are to be enforced other than control saturation.

    '''

    def __init__(self, pb, sigma, alpha):

        # pb: is the user-define object that is specific to the problem
        #       (see the provided example file: user_defined_pvtol.py)

        # Process the problem's data and functions
        #-------------------------------------------
        ## The ODE governing the system
        self.ode = pb.ode

        ## The stage constraints other than the control saturation
        self.constraints = pb.constraints
        self.final_penalty = pb.final_penalty
        self.stage_cost = pb.stage_cost

        ## dimensions of the state and the control vectors
        self.nx = pb.nx
        self.nu = pb.nu
        self.nc = pb.nc

        ## Control saturation
        self.u_min = pb.u_min
        self.u_max = pb.u_max

        ## number of parameters in the model
        self.n_p = pb.n_p

        ## number of parameters to define the cost
        self.n_q = pb.n_q

        # Process the sigma related information
        # -------------------------------------------
        ## sampling period, updating period & prediction horizon
        self.tau = pb.tau
        self.kappa = sigma.kappa(alpha)
        self.tau_u = sigma.kappa(alpha) * pb.tau
        self.n_steps = sigma.n_steps(alpha)
        self.N_prediction = sigma.N_pred(alpha)
        self.max_iter = sigma.max_iter(alpha)
        self.n_ctr = len(sigma.I_dec(alpha))

        ## final penalty maps
        self.rho_final = sigma.rho_final(alpha)

        ## Compute the decision instants and the number of dof
        # The decision instants over the prediction
        # horizon used in the control parameterization
        # The index 0 is mandatory and is not be included
        # the list of free-value instants by construction
        # (see the Sigma class for more details)

        self.I_dec = sigma.I_dec(alpha)
        if self.I_dec[-1] < self.N_prediction:
            self.n_dof = self.nu * (len(self.I_dec) + 2)
        else:
            self.n_dof = self.nu * (len(self.I_dec) + 1)

        # penalty on the soft-constraints violation
        self.rho_constr = sigma.rho_constr(alpha)

        # Induced constraints on the decision variable as
        # derived from the control profile's parameterization
        # self.nz = self.n_dof !

        self.z_min = np.array(list(self.u_min) * int(self.n_dof / self.nu))
        self.z_max = np.array(list(self.u_max) * int(self.n_dof / self.nu))
        self.nz = len(self.z_min)

        # The Casadi cost function and its gradients
        # In the present optimization related choices
        # these gradients are not used. They are included
        # for future extensions by the authors or other
        # persons willing to extend the possibilities.
        # -------------------------------------------

        ## Create the J function
        x0 = MX.sym('x', self.nx)
        p = MX.sym('p', self.n_p)
        q = MX.sym('q', self.n_q)
        z = MX.sym('z', self.n_dof)
        J = self.cost(z, x0, p, q)
        self.J = Function('J', [z, x0, p, q], [J])

        ## Create the gradient function w.r.t z

        Gz = jacobian(J, z)
        self.Gz = Function('Gz', [z, x0, p, q], [Gz])

        ## Create the cost function w.r.t x0
        Gx0 = jacobian(J, x0)
        self.Gx0 = Function('Gx0', [z, x0, p, q], [Gx0])

        # The optimization problem and the solver
        # -------------------------------------------
        ## Define the optimization problem
        self.ocp = {'f': J, 'x':z, 'p':vertcat(x0, p, q)}

        ## Define the solver
        self.solver = nlpsol('solver', 'ipopt', self.ocp,
                             {'ipopt':{'max_iter': self.max_iter}})
    # --------------------------------------------------
    def generate_z0(self):

        # Generate a random initialization for the solver
        z0 = self.z_min + np.random.rand(self.nz) * (self.z_max -self.z_min)
        return z0
    # --------------------------------------------------
    def step_ahead(self, x, u, p, q, h, nStep):

        # performs a time step (h) by subdividing the
        # computation over nStep sub-intervals.
        # when called by the simulator (h=tau, nStep=1)
        # when called by the controller, (h=tau_u, nStep=n_steps)
        # is used for the prediction.
        #-----
        # returns:
        #
        #   xc: The next state after (h) time units
        #   Jc: The cumulated stage cost augmented by the cumulated constraints penalty
        #   max_cstr: The maximum positive values of the constraints.
        #
        # Note that max_cstr is not multiplied by the penalty rho_cstr and does not
        # cumulate the contraint violation, rather, it returns the maximum violation
        # over the list of constraints and over the nStep involved in this one_step_ahead
        # prediction. This means that a threshold compared to unity is relevant for
        # this variable.

        xc = copy.copy(x)
        Jc = 0
        J_cstr, max_cstr = 0, 0

        hs = h / nStep
        for i in range(nStep):

            k1 = self.ode(xc, u, p)
            k2 = self.ode(xc + 0.5 * hs * k1, u, p)
            k3 = self.ode(xc + 0.5 * hs * k2, u, p)
            k4 = self.ode(xc + hs * k3, u, p)
            xc += hs/6.0*(k1+2*(k2+k3)+k4)
            Jc += self.stage_cost(xc, u, p, q) * hs
            if self.constraints != None:
                c = self.constraints(xc, u, p, q)
                for ic in range(self.nc):
                    term = if_else(c[ic] > 0, c[ic], 0)
                    J_cstr += self.rho_constr * (term * term)
                    max_cstr = if_else(term > max_cstr, term, max_cstr)

        Jc += J_cstr

        return xc, Jc, max_cstr
    #--------------------------------------------------
    def reconstruct_U(self, z):

        # Reconstructs the control profile over the prediction
        # horizon given the vector of parameters z
        # (decision variable for the optimizer) and the
        # list of decision instants defined by self.I_dec
        # This precisely what is called the control parameterization.

        # The first decision instant is necessarily 0
        U = vertcat(z[0:self.nu])

        for i in np.arange(1, self.N_prediction):

            # if i is beyond the last chosen decision instant,
            # take the last nu components of z
            if i >= self.I_dec[-1]:
                U = vertcat(U, z[-self.nu:])
            # if this is a decision instant
            # take the corresponding value of z
            elif i in self.I_dec:
                iz = list(self.I_dec).index(i)
                U = vertcat(U, z[(iz+1)*self.nu:(iz+2)*self.nu])
            # else: interpolate
            else:
                i0 = [j for j in self.I_dec if j <= i][-1]
                i1 = [j for j in self.I_dec if j >= i][0]
                iz0  = self.I_dec.index(i0)
                iz1 = self.I_dec.index(i1)
                u_previous = z[(iz0+1)*self.nu:(iz0+2)*self.nu]
                u_next = z[(iz1+1) * self.nu:(iz1 + 2) * self.nu]
                u_current = u_previous + (i-i0)/(i1-i0) * (u_next-u_previous)
                U = vertcat(U, u_current)
        return U
    # --------------------------------------------------
    def cost(self, z, x0, p, q):

        # The cost function including stage cost, terminal penalty
        # and the penalty on the constraints' violation.

        # Reconstruct the conrol input profile
        U = self.reconstruct_U(z)

        # Initialization
        X, xact, J = x0, x0, 0

        for i in range(self.N_prediction):

            u = U[i*self.nu:(i+1)*self.nu]

            # Recall that J_act includes the constraints violation sum multiplied
            # by the penalty weight rho_cstr.
            xact, J_act, _ = self.step_ahead(xact, u, p, q, self.tau_u, self.n_steps)
            J += J_act
            X = vertcat(X, xact)

        J /= self.N_prediction

        # Add the terminal penalty
        if self.final_penalty != None:
            J += self.rho_final * self.final_penalty(xact, p, q)

        return J
    # --------------------------------------------------
    def feedback(self, x, p, q, z0):

        # This is the NMPC feedback!
        # returns:
        #
        # z_opt: The optimal vector of decision variables
        # J_opt: The optimal value of the cost funciton.
        #
        # Recall that the cost function here regroups the integrated
        # stage cost, the integrated weighted constraint violation and
        # the final cost.

        silence_casadi_output()
        sol = self.solver(x0=z0, lbx=self.z_min, ubx=self.z_max,
                          p=vertcat(x, p, q))
        sys.stdout = sys.__stdout__

        z_opt = sol['x']
        J_opt = sol['f']
        return z_opt, J_opt
    # --------------------------------------------------
    def sim_cl(self, sc, z0=None, optim_par=None):

        # sc: is the scenario including: x0, p, q, T
        # gam: is the required contraction
        # dev_acc: The acceleration ratio of the targeted device.
        # c_max: accepted level of constraints violation.
        # --------------------------------------------------------

        # This is the utility that determines the assessment of
        # the MPC associated to a given (sigma, alpha) paire
        # (used in the instantiation of the instance (self) of
        # the MPC class when used in the scenario sc(x0,p,q,T)
        # Recall that optim_par.T determines the m-steps that defines
        # the length of the scenario in terms of updating periods.

        gam = optim_par.gam
        dev_acc = optim_par.dev_acc
        c_max = optim_par.c_max
        T = optim_par.T

        t_init = tt()
        # Extract the data from the scenario
        x0, p, q = sc.x0, sc.p, sc.q
        Nsim = int(np.round(optim_par.T/self.tau_u))

        # Initial guess of the decision variable.
        xact = x0
        if z0 is None:
            self.z0 = self.generate_z0()
        else:
            self.z0 = z0
        z_warm = self.z0

        # initialize the container for the output
        tcl = np.array([i * self.tau for i in range(Nsim * self.kappa + 1)])

        Xcl, Ucl, Jcl, cpu = [x0], [], [], []

        # Initialize the closed loop constraint violation and cost.
        cl_constr, cl_cost = [], []

        # RT compatibility certificate (a priori ok)
        # Becomes false if at some instant, the computation on the target device
        # exceed the updating period tau_u

        RT_compatible = True
        cstr_compatible= True

        for i in range(Nsim):
            t0 = tt()
            # Here is where the MPC computation is done
            # Abort the loop if the computation time exceed
            # The updating period tau_u = self.dt * self.kappa
            z_opt, J_opt = self.feedback(xact, p, q, z_warm)
            current_cpu = tt()-t0

            # check RT-compatibility
            # Notice how the device performance is taken into
            # account in the assessment of RT-compatibility
            if current_cpu / dev_acc > self.tau_u:
                RT_compatible = False
                break

            # if still RT-compatible:
            cpu += [current_cpu]
            Jcl += [J_opt.full().flatten()[0]]
            z_opt = z_opt.full().flatten()

            # Extract the first control (u) to be applied during the updating period
            # which represent kappa basic sampling periods tau.
            u = z_opt[0:self.nu]
            for _ in  range(self.kappa):
                Ucl += [u]
                xact, J_tot, max_cstr = self.step_ahead(xact, u, p, q, self.tau, 1)

                # Check the constraint satisfaction
                if max_cstr > c_max:
                    cstr_compatible = False
                    break


                # If still ok, append the results to
                cl_cost.append(np.array(J_tot).flatten()[0])
                cl_constr.append(max_cstr)
                Xcl += [xact.full().flatten()]

            # update the warm start
            z_warm =np.array(list(z_opt[self.nu:])+list(z_opt[-self.nu:]))



        # Save the results in the Container R
        # Different output are possible depending on the
        # success or failure of the MPC setting on the scenario
        #-------------------------------------------------------
        cpu_sim = tt() - t_init

        if all([RT_compatible, cstr_compatible]) :

            R = Container()
            i_end = np.argmin(abs(tcl-optim_par.T))
            R.t, R.X = tcl[0:i_end], np.array(Xcl).reshape(-1, self.nx)[0:i_end]
            R.U = np.array(Ucl).reshape(-1, self.nu)[0:i_end]
            R.J, R.cpu = np.array(Jcl), np.array(cpu)
            R.q = q
            R.cl_constr = np.array(cl_constr).flatten()[0:i_end]
            R.C_RT = R.cpu.max()/(dev_acc * self.tau_u)-1
            R.C_gam = R.J[-1]/R.J[0]-gam
            R.C_cstr = R.cl_constr.max()
            R.cl_cost = np.array(cl_cost)[0:i_end].mean()
            R.RT_compatible = RT_compatible
            R.cstr_compatible = cstr_compatible

            dic = {
                'Npred': self.N_prediction,
                'Tpred': self.N_prediction * self.tau_u,
                'tau_u': self.tau_u,
                'n_ctr': self.n_ctr,
                'kappa': self.kappa,
                'n_step': self.n_steps,
                'T_sim': R.t.max(),
                'C_RT': R.C_RT,
                'C_gam': R.C_gam,
                'C_cstr': R.C_cstr,
                'cl_cost': R.cl_cost,
                'cpu_sim_cl': cpu_sim,
                'RT_compatible': RT_compatible,
                'cstr_compatible': cstr_compatible
            }
            R.df = pd.DataFrame(dic, index=['Indicators'])

        else:

            R = Container()
            R.RT_compatible = RT_compatible
            R.cstr_compatible = cstr_compatible
            dic = {
                'Npred': self.N_prediction,
                'Tpred': self.N_prediction * self.tau_u,
                'tau_u': self.tau_u,
                'n_ctr': self.n_ctr,
                'kappa': self.kappa,
                'n_step': self.n_steps,
                'T_sim': None,
                'C_RT': None,
                'C_gam': None,
                'C_cstr': None,
                'cl_cost': None,
                'cpu_sim_cl': cpu_sim,
                'RT_compatible': RT_compatible,
                'cstr_compatible': cstr_compatible
            }
            R.df = pd.DataFrame(dic, index=['Indicators'])

        return R
    # --------------------------------------------------
    def plot_cl(self, R, fig_size=(18, 10), font_size=18, ticks_size=14, ind_x=None, ind_q=None):

        # plot the simulation result for a scenario and a specific NMPC
        # R: The output of the MPC.sim_cl method for this
        #    instance of the MPC class.
        # ind_x: The indices of the states components to plot
        # ind_q: The indices of q-components to plot

        plt.figure(figsize=fig_size)

        # Plot the evolutions of the states indexed by ind_x
        # and the components of q indexed by ind_q
        plt.subplot(3, 2, 1)
        if ind_x == None:
            ind_x = [i for i in range(self.nx)]
        plt.plot(R.t, R.X[:, ind_x])
        plt.title('Regulated output', size=font_size)
        plt.xlim([R.t.min(), R.t.max()])
        if ind_q == None:
            ind_q = [i for i in range(self.n_q)]
        plt.plot(R.t, np.ones(len(R.t)).reshape(-1, 1).dot(R.q[ind_q].reshape(1, len(ind_q))), '--')
        plt.legend(['x' + str(i + 1) for i in ind_x]+['q'+str(i+1) for i in ind_q])
        plt.tick_params(axis='both', labelsize=ticks_size)
        plt.grid(True)

        # Plot the evolution of the cost function of the optimization problem
        # during the closed-loop simulation
        plt.subplot(3, 2, 2)
        plt.semilogy(R.t[0:-1][::self.kappa][0:len(R.J)], R.J, '-o')
        plt.title('Cost function (OL)', size=font_size)
        plt.xlim([R.t.min(), R.t.max()])
        plt.tick_params(axis='both', labelsize=ticks_size)
        plt.grid(True)

        # Plot the evolution of the closed-loop control
        # together with the saturation levels for the control.
        plt.subplot(3, 2, 3)
        plt.step(R.t, R.U)
        plt.legend(['u' + str(i + 1) for i in range(self.nu)])
        plt.step(R.t, np.ones((len(R.t), 1)) * self.u_min.reshape(1, -1), '-.')
        plt.step(R.t, np.ones((len(R.t), 1)) * self.u_max.reshape(1, -1), '-.')
        plt.title('Control input', size=font_size)
        plt.xlim([R.t.min(), R.t.max()])
        plt.tick_params(axis='both', labelsize=ticks_size)
        plt.grid(True)

        # Plot the evolution of the constraint violation (maximum)
        # over the closed-loop trajectory.
        plt.subplot(3, 2, 4)
        plt.step(R.t, R.cl_constr)
        plt.title('Constraints satisfaction', size=font_size)
        plt.xlim([R.t.min(), R.t.max()])
        plt.tick_params(axis='both', labelsize=ticks_size)
        plt.grid(True)

        # Plot the evolution of the computation time over the closed-loop
        # simulation updating events.
        plt.subplot(3, 1, 3)
        t_cpu = R.t[::self.kappa]
        plt.step(t_cpu, R.cpu, lw=2, c='black')
        plt.axhline(y=self.tau_u, xmin=t_cpu.min(), xmax=t_cpu.max())
        plt.title(f'cpu / tau={self.tau:.4f} / tau_u={self.tau_u:.4f}', size=font_size)
        plt.xlim([t_cpu.min(), t_cpu.max()])
        plt.tick_params(axis='both', labelsize=ticks_size)
        plt.grid(True)
        plt.show()
    # --------------------------------------------------
    def show(self, label=None):

        # returns a dataframe summarizing the NMPC settings
        df = pd.DataFrame({
            'kappa': self.kappa,
            'N_pred': self.N_prediction,
            'n_steps': self.n_steps,
            'n_ctr':self.n_ctr,
            'rho_cstr': self.rho_constr,
            'rho_final': self.rho_final,
            'tau': self.tau,
            'tau_u': self.tau_u,
            'max_iter':self.max_iter
        }, index=[label])

        return df
def evaluate_alpha(alpha, pb, sigma, A_ell, optim_par=None, mode='middle'):

    # Evaluate the consequence of using the value alpha for a given scenario sc
    # and a given value of the shaping parameter sigma.
    # This function is used in the dichotomy that a feasible solution to the
    # problem P(sigma | A_ell) to recover hat{alpha} (if any)

    # Inputs:
    #   alpha: in (0,1) that determines (via sigma) the NMPC setting
    #   pb: The use-defined map that is problem specific.
    #   sigma: the instance of the shaping maps
    #   A_ell: The set of scenarios under consideration
    #   optim_par: the structure that defines the choice of dev_acc, gam, c_max and T
    #   mode: defines the mode used to choose the initial guess.
    #           ('middle', 'random', 'zeros')

    # returns:
    #   Summary of constraints satisfaction, dataframe and cost function
    #--------------------------------------------------------------------


    # Create the instance of class MPC
    mpc = MPC(pb, sigma, alpha)

    # compute the initial guess z0
    if mode == 'zeros':
        z0 = 0 * mpc.generate_z0()
    elif mode == 'middle':
        z0 = (mpc.z_min + mpc.z_max) / 2
    else:
        z0 = mpc.generate_z0()

    # simulate the closed loop for all the instance of A_ell
    # create the associated dataframe of results.
    df_A_ell = []
    for i in range(len(A_ell.x0)):
        x0, p, q = A_ell.x0[i], A_ell.p[i], A_ell.q[i]
        sc = Scenario(x0, p, q)
        R = mpc.sim_cl(sc, z0, optim_par)
        df_A_ell.append(R.df)

    df_A_ell = pd.concat(df_A_ell)
    df_A_ell.reset_index(drop=True, inplace=True)

    # check whether the setting is real-time compatible for all the
    # scenarios in the set A_ell
    real_time = all(df_A_ell['RT_compatible'].values)

    # Note that if the simulation stops because of RT compatibility
    # then, the satisfaction of the constraints cannot be
    # determined for a scenario.

    if real_time:
        constraint = all(df_A_ell['cstr_compatible'].values)
    else:
        constraint = None

    # Only if the simulation terminated that one can check the
    # contraction constraints.
    if all([real_time, constraint]):
        contraction = df_A_ell['C_gam'].values.max()
    else:
        contraction = None

    # only if everything is ok that the cost value matters.
    if all([real_time, constraint, contraction]):
        cl_cost = df_A_ell['cl_cost'].values.mean()
    else:
        cl_cost = None

    summary = Container()
    summary.real_time = real_time
    summary.constraint = constraint
    summary.contraction = contraction
    summary.cl_cost = cl_cost
    summary.df = df_A_ell
    summary.ok = all([summary.real_time,
                      summary.contraction,
                      summary.constraint])

    return summary

def find_alpha_max(pb, sigma, A_ell,
                   optim_par, mode='middle',
                   alpha_precision=0.1,
                   alpha_0=0.0,
                   alpha_1=1.0):

    # Determines the maximum value of alpha that is compatible with the
    # constraints. If no such alpha exists, this is mentioned through
    # the information J_out = np.inf
    # ------------------------------------------------------------------

    # Evaluate the extreme values 0, 1

    S0 = evaluate_alpha(alpha_0, pb, sigma, A_ell, optim_par, mode=mode)
    S1 = evaluate_alpha(alpha_1, pb, sigma, A_ell, optim_par, mode=mode)

    # If even for alpha=0 RT constraint is violated stop
    if not S0.real_time:
        alpha_out = None
        J_out = np.inf
    else:
        alpha_out = alpha_0
        J_out = S0.cl_cost
        if S1.real_time:
            alpha_out = alpha_1
            J_out = S1.cl_cost
        # start dichotomy
        else:
            alpha_out = alpha_0
            J_out = S0.cl_cost
            alpha_min = alpha_0
            alpha_max = alpha_1
            while alpha_max - alpha_min > alpha_precision:
                alpha = 0.5 * (alpha_min + alpha_max)
                print(f'Dichotomy on: {[alpha_min, alpha_max]}')
                S_alpha = evaluate_alpha(alpha, pb, sigma, A_ell, optim_par, mode)
                if S_alpha.real_time:
                    alpha_min = alpha
                    J_out = S_alpha.cl_cost
                else:
                    alpha_max = alpha
            alpha_out = alpha_min

    if J_out is None:
        J_out = np.inf

    return alpha_out, J_out





    return S

def generate_A(pb, n_batch, n_sc_batch):

    # Generates the set of scenario batched through n_batch subsets A_ell each
    # having n_sc_batch scenario. Notice that this utility calls the user defined
    # function generate cloud that should be provided by the user.
    # (see the user_defined_pvtol.py example)

    A = []
    for i in range(n_batch):
        A_ell = pb.generate_cloud(nSamples=n_sc_batch)
        A.append(A_ell)
    return A

def Design_MPC(pb, set_of_sigma, A, optim_par, mode='zero'):

    sys.stdout = sys.__stdout__

    bad_list = []
    n_excluded = []
    selected_alpha = {j: None for j in range(len(set_of_sigma))}
    selected_mpc = {j: None for j in range(len(set_of_sigma))}
    cumulated_cost = {j: 0 for j in range(len(set_of_sigma))}
    for i_tr in range(len(A)):
        Results = []
        for j in range(len(set_of_sigma)):
            if j not in bad_list:
                print("-------------")
                print(f"inspecting config #{j} for the set A[{i_tr}]")
                print("-------------")
                sigma = set_of_sigma[j]
                if selected_alpha[j] is None:
                    alpha_max, J_opt = find_alpha_max(pb, sigma, A[i_tr],
                                                      optim_par, alpha_precision=optim_par.eps,
                                                      alpha_0=0, alpha_1=1, mode=mode)
                else:
                    alpha_max, J_opt = find_alpha_max(pb, sigma, A[i_tr],
                                                      optim_par, alpha_precision=optim_par.eps,
                                                      alpha_0=selected_alpha[j],
                                                      alpha_1=selected_alpha[j], mode=mode)
                if J_opt != np.inf:
                    selected_alpha[j] = alpha_max
                    selected_mpc[j] = MPC(pb, sigma, alpha_max)
                    cumulated_cost[j] += J_opt

                else:
                    bad_list.append(j)


                Results.append([j, alpha_max, J_opt])
                print('--------- current situation --------')
                print(np.array(Results))
                print('------------------------------------')
        n_excluded.append(len(bad_list))

        print(f'number of removed configuration = {len(bad_list)}')


    df_mpc = pd.concat([selected_mpc[k].show(k) for
                            k in range(len(set_of_sigma))
                            if k not in bad_list])

    costs = [cumulated_cost[k] for
             k in range(len(set_of_sigma))
             if k not in bad_list]

    df_mpc['cost'] = costs
    df_mpc['alpha'] = [selected_alpha[k] for
                       k in range(len(set_of_sigma))
                       if k not in bad_list]

    R_design_log = Container()
    R_design_log.df = df_mpc
    R_design_log.n_excluded = n_excluded

    return R_design_log
