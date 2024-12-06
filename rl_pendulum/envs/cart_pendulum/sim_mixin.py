import pandas as pd
import numpy as np
import sympy as smp
from scipy.integrate import solve_ivp
        
class Cart_Pendulum_Simulation_Mixin():
    """ Mixin class that provides simulation methods for Cart_Pendulum_Environment. 
        Needs self.math was set by Cart_Pendulum_Math before usage. 
        
        For non-RL simulations: use this class as provided.
        To do RL just use: angular_data  = solve_ivp(lambda t_,x_: 
                                   robotic_equation(*x_, df_input.loc[np.round(t_, decimals=decimals)][0], *self.ddqf_constants['sym_to_val'].values()), # having to round some t_ indexes as scipy does not return them in 4 digits
                                   t_span=[0,t[-1]], y0=initial_values, method='RK45', t_eval=t, rtol=1e-6) 
        inside the step function.
    """
    
    def full_simulation_data(self, **params):
        """ Returns full simulation trajectory, allows only non-reactive control input.
            Allows fast parallel computation for testing. 
            The control function has to be a differentiable sympy function of the x-cart-position only dependent on t. 
            Define a 0 function to disable any control input.
            - The control function should of course be chosen, so that the cart will not leave the track bounds
            - This function does not implement bumping for computational efficiency; bumping is later ignored too for RL
            - Note: parallel computation of this simulation is easily possible when ignoring bumping
                --> In order to implement bumping here correctly, one needs to implement a step-wise calculation of acceleration, velocity, and position, and apply the solver for EVERY SINGLE step, not just once. 
                --> The if statements along the step-wise integrating are slow too, and do not really allow flexible simulations for more than 10000 steps.
                --> Also consider using \ddot{x_c} as control function input, not x_c
        """
        
        assert (len(params['initial_angles'])     == self.n)
        assert (len(params['initial_velocities']) == self.n) 
        assert (params['dt'] < 0.5) 
        
        T = params['T']
        dt = params['dt'] # like 0.001
        steps = int(np.ceil(T/dt))
        s = str(dt)
        decimals = len(s[s.find('.')+1:])
        t = np.linspace(0, T, steps+1)
        t = np.round(t, decimals=decimals) # avoiding 1.250000001 s
        initial_angles     = params['initial_angles']
        initial_velocities = params['initial_velocities']
        control_func       = params['control_function']
        control_func_xc    = smp.lambdify([self.variables['t']], [control_func]) # (just for data analysis)
        control_func_dxc   = smp.lambdify([self.variables['t']], [control_func.diff(self.variables['t'], 1)]) # position to velocity (just for data analysis)
        control_func_ddxc  = smp.lambdify([self.variables['t']], [control_func.diff(self.variables['t'], 2)]) # position to acceleration
        robotic_equation   = self.math['str_to_sol_func']['ddq_f']
        initial_values = initial_angles + initial_velocities # list appending
        
        # calculating additional x_cart values
        if control_func != 0:
            xc_data   = np.array(control_func_xc(t))
            dxc_data  = np.array(control_func_dxc(t))
            ddxc_data = np.array(control_func_ddxc(t))
        elif control_func == 0:
            xc_data   = np.zeros((1, steps+1))
            dxc_data  = np.zeros((1, steps+1))
            ddxc_data = np.zeros((1, steps+1)) 
            
        df_input = pd.DataFrame(ddxc_data[0], index = t) # allowing for t_ indexing
        
        angular_data  = solve_ivp(lambda t_,x_: 
                                  robotic_equation(*x_, df_input.loc[np.round(t_, decimals=decimals)][0], *self.ddqf_constants['sym_to_val'].values()), # having to round some t_ indexes as scipy does not return them in 4 digits
                                  t_span=[0,t[-1]], y0=initial_values, method='RK45', t_eval=t, rtol=1e-6) 
        
        xc_data_  = np.concatenate((xc_data, dxc_data, ddxc_data), axis=0)
        
        theta_data  = angular_data['y'][       : self.n]
        dtheta_data = angular_data['y'][self.n : self.n*2]
        ddtheta_data = np.diff(dtheta_data)/dt
        ddtheta_data = np.concatenate(([[0]]*self.n, ddtheta_data), axis=1) # [[0]]*n --> [[0], [0], [0]]
        
        # calculating rod x-y positions
        str2sym           = self.math['str_to_sym']
        str2func          = self.math['str_to_func']
        q_func2sym        = self.substitution_mappings['q_func2sym']
        
        l_strings = [f"l_{n_}" for n_ in self.n_range]
        l_symbols = [self.constants['str_to_sym'][str_] for str_ in l_strings]
        l_values  = [self.constants['sym_to_val'][sym]  for sym  in l_symbols]
    
        x_c_strings       = ["x_c", r"\dot{x_c}", r"\ddot{x_c}"]
        theta_strings     = [fr"\theta_{n_}"  for n_ in self.n_range]
        dtheta_strings    = [fr"\dot{{\theta_{n_}}}"   for n_ in self.n_range]
        ddtheta_strings   = [fr"\ddot{{\theta_{n_}}}"  for n_ in self.n_range]   
        
        x_c_symbols       = [str2sym[str_] for str_ in     x_c_strings]
        theta_symbols     = [str2sym[str_] for str_ in   theta_strings]
        dtheta_symbols    = [str2sym[str_] for str_ in  dtheta_strings]
        ddtheta_symbols   = [str2sym[str_] for str_ in ddtheta_strings]
        
        pxr_strings       = [fr"p_{{x_{n_}}}^{{r}}" for n_ in self.n_range]
        pyr_strings       = [fr"p_{{y_{n_}}}^{{r}}" for n_ in self.n_range]

        symbolic_funcs_pxr = [str2func[str_].subs(q_func2sym) for str_ in pxr_strings] # we cant lambdify a function x(t) both with x and t --> reduce x_n(t) func to x_n symbol
        symbolic_funcs_pyr = [str2func[str_].subs(q_func2sym) for str_ in pyr_strings]
        pxr_functions     = [smp.lambdify([x_c_symbols[0]] + theta_symbols[:n_] + l_symbols[:n_], [symbolic_funcs_pxr[n_-1]]) for n_ in self.n_range]
        pyr_functions     = [smp.lambdify(                   theta_symbols[:n_] + l_symbols[:n_], [symbolic_funcs_pyr[n_-1]]) for n_ in self.n_range]
        
        l_constant_data   = np.tile(l_values, (steps+1, 1)).T

        pxr_data          = np.array([pxr_functions[n_-1](*np.concatenate([xc_data, theta_data[:n_,:], l_constant_data[:n_,:]], axis=0)) for n_ in self.n_range]) # have to put '*' before nested lists (more than 1 feature series)! A big error source
        pyr_data          = np.array([pyr_functions[n_-1](*np.concatenate([         theta_data[:n_,:], l_constant_data[:n_,:]], axis=0)) for n_ in self.n_range])
        pxr_data          = np.squeeze(pxr_data, axis=1) # (n, 1, steps) --> (n, steps)
        pyr_data          = np.squeeze(pyr_data, axis=1)
        data              = np.concatenate((xc_data_, pxr_data, pyr_data, theta_data, dtheta_data, ddtheta_data), axis=0)
        df = pd.DataFrame(data.T, index=t,
                          columns=['$$x_c$$', '$$\dot{x_c}$$', '$$\ddot{x_c}$$'] + 
                                  [fr"$$p_{{x_{n_}}}^{{r}}$$" for n_ in self.n_range] +
                                  [fr"$$p_{{y_{n_}}}^{{r}}$$" for n_ in self.n_range] +
                                  [fr'$$\theta_{{{i}}}$$' for i in self.n_range] +
                                  [fr'$$\dot{{\theta_{i}}}$$' for i in self.n_range] +
                                  [fr'$$\ddot{{\theta_{i}}}$$' for i in self.n_range])
        df.index.name = 't'
        return df