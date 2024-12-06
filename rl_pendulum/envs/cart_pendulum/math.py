import pandas as pd
import numpy as np
import sympy as smp
from scipy.integrate import odeint, solve_ivp

            
class Cart_Pendulum_Math():
    """ This is a Mixin class inside object-oriented programming (OOP). 
        The math was used from https://colab.research.google.com/drive/1tonlB7P0w4EZv2eC8PMP9zO-FzwBizb_#scrollTo=GwZi4egvspyT"""
    
    """ # Note: Initializing this class will take more than a minute for n>3
        ## Slow Sympy functions like smp.simplify() do not support multiprocessing and are always executed on one core.
    """
    def __init__(self, **constants): 
        self._define_constants(constants)
        self._define_variables()
        self._define_simulation_functions()
        self._define_q_matrices_and_q_mappings()
        self._define_equations_and_matrices()

    def _define_constants(self, constants_str2val):
        # Defining constant string-to-symbol mapping and symbol-to-value mapping here
        consts_s  = ['n'] + ['g'] + ['w'] + ['w_c'] + ["m_c"]# n: number of rods, w: track width, w_c: cart width, m_c: cart mass (no effect inside simulation!!)
        l_s       = [f"l_{n_}"                for n_ in self.n_range] # rod lengths
        r_s       = [f"r_{n_}"                for n_ in self.n_range] # mass point distance to rod origin
        m_s       = [f"m_{n_}"                for n_ in self.n_range] # rod masses
        mu_s      = [r"\mu_0"] + [fr"\mu_{n_}" for n_ in self.n_range] # rod joint friction coefficients
        I_s       = [f"I_{n_}"                for n_ in self.n_range] # moments of inertia for each rod (=1/3*mi*r_i^2)
        
        consts_v  = [constants_str2val[str_]        for str_ in  consts_s]
        l_v       = [constants_str2val['l'][i-1]    for i in self.n_range]
        r_v       = [constants_str2val['r'][i-1]    for i in self.n_range]
        m_v       = [constants_str2val['m'][i-1]    for i in self.n_range]
        mu_v      = [constants_str2val[r'\mu_c']] + list(constants_str2val[r'\mu']) # --> [mu_c, mu_1, ..., mu_n]; list() for converting np.array to list
        I_v       = [constants_str2val['I'][i-1]    for i in self.n_range]
        constants = consts_s + l_s + r_s + m_s + mu_s + I_s
        values    = consts_v + l_v + r_v + m_v + mu_v + I_v
        constants_str_to_sym = {str_ : smp.symbols(str_, real=True, positive=True) for str_ in constants}  
        constants_str_to_val = {str_ : val for str_, val in zip(constants, values)}  
        constants_sym_to_val = {sym  : val                                         for sym, val in zip(constants_str_to_sym.values(), values)}   
        
        self.constants = {'str_to_sym' : constants_str_to_sym, 
                          'str_to_val' : constants_str_to_val, 
                          'sym_to_val' : constants_sym_to_val} # simply holds all constants
        
        consts_ddqf_s  = ['g'] # only needing g for ddqf (the final equation of motion)
        consts_ddqf_v  = [constants_str2val[str_] for str_ in  consts_ddqf_s]
        ddqf_constants = consts_ddqf_s + l_s[:-1] + r_s + m_s + mu_s[1:] + I_s # list appending; last l ist not needed!; \mu_0 not needed!
        ddqf_values    = consts_ddqf_v + l_v[:-1] + r_v + m_v + mu_v[1:] + I_v
        ddqf_constants_str_to_sym = {str_ : smp.symbols(str_, real=True, positive=True) for str_ in ddqf_constants}  
        ddqf_constants_str_to_val = {str_ : val                                         for str_, val in zip(ddqf_constants, ddqf_values)}  
        ddqf_constants_sym_to_val = {sym  : val                                         for sym, val in zip(ddqf_constants_str_to_sym.values(), ddqf_values)}   
        
        self.ddqf_constants = {'str_to_sym' : ddqf_constants_str_to_sym, 
                               'str_to_val' : ddqf_constants_str_to_val, 
                               'sym_to_val' : ddqf_constants_sym_to_val} # only holds all constants needed inside ddqf (the final equation of motion)
                                                                         # having n or w substituted into ddqf results in broken pendulum danamics...
    def _define_variables(self):
        # (Changed with environment step, impacts functions)
        t, a_a    = [['t'], ['a_a']]
        variables = t + a_a
        self.variables = {sym : smp.symbols(sym, real=True) for sym in variables}
        
    def _define_simulation_functions(self):
        
        # Lists to be filled for string-symbol-function mappings inside self.math
        sim_func_strings       = []
        sim_func_expressions   = []
        sim_matrix_strings     = []
        sim_matrix_funcs       = []
        
        # neccessary environment functions:
        ## Cart functions
        x_c_strings          = ["x_c", r"\dot{x_c}", r"\ddot{x_c}"]
        d_left_string        = ["d_left"]
        d_right_string       = ["d_right"]
        x_c_functions        = [smp.Function(x_c_str_, real=True)(self.variables['t']) for x_c_str_ in x_c_strings]
        x_c_f                = x_c_functions[0]
        d_left_f             = smp.Rational(1,2)*self.constants['str_to_sym']['w'] + x_c_f - smp.Rational(1,2)*self.constants['str_to_sym']['w_c']
        d_right_f            = smp.Rational(1,2)*self.constants['str_to_sym']['w'] - x_c_f - smp.Rational(1,2)*self.constants['str_to_sym']['w_c']
        sim_func_strings     += x_c_strings      + d_left_string + d_right_string
        sim_func_expressions += x_c_functions  + [d_left_f]    + [d_right_f]
        
        ## Joint Angles Θ
        theta_strings        = [fr"\theta_{n_}"  for n_ in self.n_range]
        dtheta_strings       = [fr"\dot{{\theta_{n_}}}"   for n_ in self.n_range]
        ddtheta_strings      = [fr"\ddot{{\theta_{n_}}}"  for n_ in self.n_range]
        theta_functions      = [smp.Function(str_, real=True)(self.variables['t']) for str_ in   theta_strings] 
        dtheta_functions     = [smp.Function(str_, real=True)(self.variables['t']) for str_ in  dtheta_strings]
        ddtheta_functions    = [smp.Function(str_, real=True)(self.variables['t']) for str_ in ddtheta_strings]
        sim_func_strings     += theta_strings   + dtheta_strings   + ddtheta_strings
        sim_func_expressions += theta_functions + dtheta_functions + ddtheta_functions
        
        ## Rod Tip Positions; x,y positions
        #### might be used for joint position analysis; (Center of Mass Positions are used for simulating)
        pxr_strings          = [f"p_{{x_{n_}}}^{{r}}"                                                   for n_ in self.n_range]
        pyr_strings          = [f"p_{{y_{n_}}}^{{r}}"                                                   for n_ in self.n_range]
        pxr_summands         = [-self.constants['str_to_sym'][f"l_{n_}"]*smp.sin(theta_functions[n_-1]) for n_ in self.n_range]
        pyr_summands         = [self.constants['str_to_sym'][f"l_{n_}"]*smp.cos(theta_functions[n_-1])  for n_ in self.n_range]           
        pxr_funcs            = [sum(pxr_summands[:n_]) + x_c_f                                          for n_ in self.n_range]
        pyr_funcs            = [sum(pyr_summands[:n_])                                                  for n_ in self.n_range]
        sim_func_strings     += pxr_strings + pyr_strings
        sim_func_expressions += pxr_funcs + pyr_funcs
        
        ## Center of Mass Positions; x,y positions 
        ### The formulas now use length r instead of l for only the last summand
        pxm_strings          = [f"p_{{x_{n_}}}^{{m}}"                                                                                      for n_ in self.n_range]  
        pym_strings          = [f"p_{{y_{n_}}}^{{m}}"                                                                                      for n_ in self.n_range]
        pxm_summands         = [-self.constants['str_to_sym'][f"l_{n_}"]*smp.sin(theta_functions[n_-1])                                    for n_ in self.n_range] # same as above for pxr
        pym_summands         = [self.constants['str_to_sym'][f"l_{n_}"]*smp.cos(theta_functions[n_-1])                                     for n_ in self.n_range] # same as above for pyr
        pxm_funcs            = [sum(pxm_summands[:n_-1]) - self.constants['str_to_sym'][f"r_{n_}"]*smp.sin(theta_functions[n_-1]) + x_c_f  for n_ in self.n_range]
        pym_funcs            = [sum(pym_summands[:n_-1]) + self.constants['str_to_sym'][f"r_{n_}"]*smp.cos(theta_functions[n_-1])          for n_ in self.n_range]
        sim_func_strings     += pxm_strings + pym_strings
        sim_func_expressions += pxm_funcs + pym_funcs
        
        ## Energy functions
        L_string             = ['L']
        T_trans_strings      = ["T_{trans_{c}}"] + [f"T_{{trans_{n_}}}" for n_ in self.n_range] # T_trans for m_c ignored for feature engineering
        T_rot_strings        = [f"T_{{rot_{n_}}}"   for n_ in self.n_range]     
        V_strings            = [f"V{n_}" for n_ in self.n_range]
        T_V_strings          = ['T_{trans}', 'T_{rot}', 'V'] 
        
        q_diff_f2dot_f       = {smp.diff(func, self.variables['t']) : d_func for func, d_func in zip([x_c_functions[0]] + theta_functions, [x_c_functions[1]] + dtheta_functions)}
        T_trans_summands     = [smp.Rational(1,2)*self.constants['str_to_sym'][f"m_c"] * x_c_functions[1]**2]
        T_trans_summands    += [smp.Rational(1,2)*self.constants['str_to_sym'][f"m_{n_}"] * (smp.diff(pxm_funcs[n_-1],   self.variables['t'])**2 + smp.diff(pym_funcs[n_-1], self.variables['t'])**2) for n_ in self.n_range]
        T_trans_summands     = [summand.subs(q_diff_f2dot_f) for summand in T_trans_summands] # see _define_q_matrices_and_q_mappings() under q_diff_f2dot_f
        T_rot_summands       = [smp.Rational(1,2)*self.constants['str_to_sym'][f"I_{n_}"] * dtheta_functions[n_-1]**2     for n_ in self.n_range] #### changed smp.diff(theta_functions[n_-1], self.variables['t'])**2 28.12              
        V_summands           = [self.constants['str_to_sym'][f"m_{n_}"]*self.constants['str_to_sym']['g']*pym_funcs[n_-1] for n_ in self.n_range]
        
        T_trans_func         = [sum(T_trans_summands).simplify()]
        T_rot_func           = [sum(T_rot_summands).simplify()] # separate simplifying for each way faster!
        V_func               = [sum(V_summands)]
        L_func               = [((T_trans_func[0] + T_rot_func[0]).simplify() - V_func[0]).simplify()]
        
        sim_func_strings     += T_V_strings                        + T_trans_strings + T_rot_strings + V_strings + L_string 
        sim_func_expressions += T_trans_func + T_rot_func + V_func + T_trans_summands + T_rot_summands + V_summands + L_func

        simulation_functions_str_to_sym  = {str_ : smp.symbols(str_, real=True) for str_       in     sim_func_strings}  
        simulation_functions_sym_to_func = {sym  : func                         for sym,  func in zip(simulation_functions_str_to_sym.values(), sim_func_expressions)}                       
        simulation_functions_str_to_func = {str_ : func                         for str_, func in zip(sim_func_strings, sim_func_expressions)}  
        
        self.math = {'str_to_sym'      : simulation_functions_str_to_sym, 
                     'sym_to_func'     : simulation_functions_sym_to_func, 
                     'str_to_func'     : simulation_functions_str_to_func,
                     'str_to_matrix'   : {}, 
                     'str_to_equation' : {},
                     'str_to_sol_func' : {}}
        
    def _define_q_matrices_and_q_mappings(self):
        
        sim_matrix_strings = []
        sim_matrix_funcs   = []
        
        # unpacking self.math
        ## Reusing code for string names from function above (dependency)
        str2func          = self.math['str_to_func']      
        str2sym           = self.math['str_to_sym']
        t                 = self.variables['t']
        
        x_c_strings       = ["x_c", r"\dot{x_c}", r"\ddot{x_c}"]
        theta_strings     = [fr"\theta_{n_}"  for n_ in self.n_range]
        dtheta_strings    = [fr"\dot{{\theta_{n_}}}"   for n_ in self.n_range]
        ddtheta_strings   = [fr"\ddot{{\theta_{n_}}}"  for n_ in self.n_range]
        
        x_c_symbols       = [str2sym[str_] for str_ in     x_c_strings]
        theta_symbols     = [str2sym[str_] for str_ in   theta_strings]
        dtheta_symbols    = [str2sym[str_] for str_ in  dtheta_strings]
        ddtheta_symbols   = [str2sym[str_] for str_ in ddtheta_strings]
    
        x_c_functions     = [str2func[str_] for str_ in     x_c_strings]
        theta_functions   = [str2func[str_] for str_ in   theta_strings]
        dtheta_functions  = [str2func[str_] for str_ in  dtheta_strings]
        ddtheta_functions = [str2func[str_] for str_ in ddtheta_strings]
        
        # Defining q matrices (q holds all q-functions)
        ## q stands generally for any fully time-dependent function 
        ## q are always the functions to be solved for in dynamical systems
        q_strings          = ['q', r'\dot{q}', r'\ddot{q}']
        q_functions        = [x_c_functions[0]] +   theta_functions
        dq_functions       = [x_c_functions[1]] +  dtheta_functions
        ddq_functions      = [x_c_functions[2]] + ddtheta_functions
        q                  = smp.Matrix(q_functions)
        dq                 = smp.Matrix(dq_functions)
        ddq                = smp.Matrix(ddq_functions)
        sim_matrix_strings += q_strings
        sim_matrix_funcs   += [q, dq, ddq]
        
        # Defining q substitution mappings:
        q_symbols       = [x_c_symbols[0]] +   theta_symbols
        dq_symbols      = [x_c_symbols[1]] +  dtheta_symbols
        ddq_symbols     = [x_c_symbols[2]] + ddtheta_symbols
        
        q_func2sym    = {func    :   sym  for    func,    sym in zip(  q_functions,   q_symbols)} # substituing symbols for fonctions (symbols are needed for ode-systems instead of functions)
        dq_func2sym   = {d_func  :  d_sym for  d_func,  d_sym in zip( dq_functions,  dq_symbols)}
        ddq_func2sym  = {dd_func : dd_sym for dd_func, dd_sym in zip(ddq_functions, ddq_symbols)} # Not needed but added in just in case 
        
        q_diff_f2dot_f      = {smp.diff(func, t)   :  d_func for   func,  d_func in zip(q_functions,   dq_functions)} # substituting \dot{q_i} for \frac{d}{dt}q_i (a lot cleaner for visualization and simplification)
        q_diff_dot_f2ddot_f = {smp.diff(d_func, t) : dd_func for d_func, dd_func in zip(dq_functions, ddq_functions)} 
        self.substitution_mappings.update({'q_func2sym'          : q_func2sym, 
                                           'dq_func2sym'         : dq_func2sym, 
                                           'ddq_func2sym'        : ddq_func2sym,
                                           'q_diff_f2dot_f'      : q_diff_f2dot_f,
                                           'q_diff_dot_f2ddot_f' : q_diff_dot_f2ddot_f})
        
        simulation_matrices_str_to_func = {str_ : func for str_, func in zip(sim_matrix_strings, sim_matrix_funcs)}
        
        self.math['str_to_matrix'].update(simulation_matrices_str_to_func)
        
    def _define_equations_and_matrices(self):
        
        # Lists to be filled:
        sim_equation_strings = []
        sim_equations        = [] # equaions are always set to 0
        sim_matrix_strings   = []
        sim_matrices         = [] # Matrices miss their arguments: 'M' instead of 'M(q)'
        sim_solution_strings = []
        sim_solution_func    = [] # Solution function gathered by lambdifying solution systems
        
        # unpacking self.math
        ## placing all naming dependencies here:
        str2func       = self.math['str_to_func']
        str2sym        = self.math['str_to_sym']
        str2mat        = self.math['str_to_matrix']
        t              = self.variables['t']
        L_func         = str2func['L']
        mu_strings     = ["\mu_0"] + [fr"\mu_{n_}" for n_ in self.n_range]
        mu_symbols     = [self.constants['str_to_sym'][str_] for str_ in mu_strings]
        ddx_c_symbol   = str2sym[r'\ddot{x_c}']
        theta_strings  = [fr"\theta_{n_}"          for n_ in self.n_range]
        dtheta_strings = [fr"\dot{{\theta_{n_}}}"  for n_ in self.n_range]
        theta_symbols  = [str2sym[str_]            for str_ in  theta_strings]
        dtheta_symbols = [str2sym[str_]            for str_ in  dtheta_strings]
        
        q   = str2mat['q']
        dq  = str2mat[r'\dot{q}']
        ddq = str2mat[r'\ddot{q}']
        
        # unpack self.substitution_mappings
        q_func2sym          = self.substitution_mappings['q_func2sym']
        dq_func2sym         = self.substitution_mappings['dq_func2sym']
        q_diff_f2dot_f      = self.substitution_mappings['q_diff_f2dot_f']
        q_diff_dot_f2ddot_f = self.substitution_mappings['q_diff_dot_f2ddot_f']
        
        # Euler Lagrange Equations for any x_c and theta functions (they contain their 1st + 2nd derivatives within smp.diff(L))
        Lode_string          = ['L_{ODE}']
        Lode                 = (smp.diff(smp.Matrix([L_func]).jacobian(dq),t) - smp.Matrix([L_func]).jacobian(q))
        Lode                 = Lode.T.subs(q_diff_f2dot_f).subs(q_diff_dot_f2ddot_f)
        Lode                 = smp.simplify(Lode)
        sim_equation_strings += Lode_string
        sim_equations        += [Lode]
        
        # Intertia Matrix M(q)
        M_string           = ['M(q)']
        M                  = smp.factor(Lode.jacobian(ddq))
        M                  = smp.simplify(M)
        sim_matrix_strings += M_string
        sim_matrices       += [M]
        
        # Lode_diff(q, \dot{q})
        ## Lode_diff is very handy to derive C and g. They can be accurately derived with the regular Lode too (see formulas).
        ## Simplifiying Lode_diff also spares a lot of time compared to simplifying Lode (for n=3: 10s instead of 50s!)
        Lode_diff_string     = ['L_{ODE,diff}']
        Lode_diff            = Lode - M*ddq
        Lode_diff            = smp.simplify(Lode_diff) # Lode_diff.simplify() does not work here! 
        sim_equation_strings += Lode_diff_string
        sim_equations        += [Lode_diff]

        # Coriolismatrix C(q, \dot{q})
        ## dM-2C should be skew-symmetric
        C_string           = ['C(q, \dot{q})']
        C                  = smp.Rational(1,2)*Lode_diff.jacobian(dq) 
        sim_matrix_strings += C_string
        sim_matrices       += [C]
        
        skew_test = (smp.diff(M, t).subs(q_diff_f2dot_f) - 2*C) + (smp.diff(M, t).subs(q_diff_f2dot_f) - 2*C).T
        skew_test = smp.simplify(skew_test)
        assert (skew_test.is_zero_matrix)
        del skew_test
        
        # Gravitational forces vector g(q)
        ## gravitational forces acting on each link
        g_string           = ['g(q)']
        g                  = Lode_diff - C*dq
        g                  = smp.simplify(g) 
        sim_matrix_strings += g_string
        sim_matrices       += [g]
        
        ode_check = (M*ddq + C*dq + g - Lode).simplify()
        assert (ode_check.is_zero_matrix)
        del ode_check
        
        # Damping Matrix D
        ## contains only joint friction coefficients on 3 diagonals (upper, main, lower diagonal) as damping only depends on relative joint movements
        D_string     = ['D']
        D_size       = self.n+1
        mu           = mu_symbols
        upper_diag   = [0] + [-mu[i] for i in self.n_range[1:]]
        main_diag    = [mu[0]] + [ mu[i] + mu[i+1] for i in self.n_range[:-1] ] + [mu[-1]]
        lower_diag   = upper_diag 
        D            = smp.Matrix(np.zeros((D_size,D_size), dtype=int))
        
        # assiging diagonal values to empty D
        for i in range(D_size-1):
            D[i,i+1] = upper_diag[i]

        for i in range(D_size):
            D[i,i]   = main_diag[i]

        for i in range(D_size-1):
            D[i+1,i] = lower_diag[i]
        
        sim_matrix_strings += D_string
        sim_matrices       += [D]
            
        # Vector of generalzed forces b(q)
        ## b is a matrix for control-induced forces (inertia forces here)
        b_string        = ['b(q)']
        b = -M[:,0]
        sim_matrix_strings += b_string
        sim_matrices       += [b]
        
        # Simplification for the pendulum
        ## - a speed controller can let the cart behave like a double integrator 
        ## - thich means that no forces from the links will affect the cart (when they are perfectly compensated by the controller)
        ## - there will only be the inertia of the links, when the cart moves
        ## - so the links do not affect the cart in any way; these parts are eliminated from the matrices
    
        ### Redefinitions ###
        
        ## Redifinition for link_ode
        ### - all link-to-cart interactions are removed here
        ### - the cart-to-link interactions stays with b - a part of the inertia matrix for cart-to-link-inertia
        q_simp   =   q[1:,0]
        dq_simp  =  dq[1:,0]
        ddq_simp = ddq[1:,0]
        M_simp   =   M[1:,1:]
        C_simp   =   C[1:,1:]
        D_simp   =   D[1:,1:]
        g_simp   =   g[1:,0] 
        b_simp   =  -M[1:,0]
        
        # simplified robotic pendulum ode
        ## equations of motion will finally be derived from here
        ## There are 2*n variables to solve for as of n second-order derivatives
        robotic_ode_string   = ['L_{ODE,robotic}']
        robotic_ode          = M_simp*ddq_simp + C_simp*dq_simp + g_simp + D_simp*dq_simp - b_simp*ddx_c_symbol # just used for math summary
        sim_equation_strings += robotic_ode_string        
        sim_equations        += [robotic_ode]
        
        ddq_equation_symbol  = [r'\hat{q}']
        ddq_sol_symbolic     = M_simp.LUsolve(-C_simp*dq_simp - g_simp - D_simp*dq_simp + b_simp*ddx_c_symbol) # symbolic solution for thetas + first derivatives
        ddq_sol_symbolic     = ddq_sol_symbolic.subs(q_func2sym).subs(dq_func2sym) # Substituting symbols instead of functions, else lambdify not working 
        sim_equation_strings += ddq_equation_symbol        
        sim_equations        += [ddq_sol_symbolic]
        
        ddq_f_solution_string = ['ddq_f']
        ddq_sol_system        = smp.Matrix([*dtheta_symbols, *ddq_sol_symbolic]) # express n second-order derivatives as 2n solvable first-order derivatives to solve for
        ddq_f                 = smp.lambdify([*theta_symbols, *dtheta_symbols, ddx_c_symbol, *self.ddqf_constants['sym_to_val']], [*ddq_sol_system]) # important to use only ddqf_constants
        sim_solution_strings  += ddq_f_solution_string
        sim_solution_func     += [ddq_f]
        
        simulation_equations_str_to_equation = {str_ : eqn   for str_,  eqn  in zip(sim_equation_strings, sim_equations)}
        simulation_matrices_str_to_matrix    = {str_ : mtrx  for str_, mtrx  in zip(sim_matrix_strings, sim_matrices)}
        simulation_solutions_str_to_solution = {str_ : sol_f for str_, sol_f in zip(sim_solution_strings, sim_solution_func)}
        
        self.math['str_to_equation'].update(simulation_equations_str_to_equation)
        self.math['str_to_matrix'  ].update(simulation_matrices_str_to_matrix)
        self.math['str_to_sol_func'].update(simulation_solutions_str_to_solution)