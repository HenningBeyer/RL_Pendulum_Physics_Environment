import numpy as np
import dill 

class Cart_Pendulum_Parameter_Manager():
    """ Class which unpacks parameters and checks them for correct definition. 
        It also pre-defines every possible attribute in advance for clarity and has no effect on other classes.
        This is the first class instantiated when instantiating Cart_Pendulum_Environment class.
        
        This class is also very useful for validity checks requested by the UI.
    """
    def __init__(self, **params):
        
        # Parameter unpacking:
        self.n                      = params['constants']['n']
        self.n_range                = np.arange(1, self.n+1) 
        self.n_plus_1_range         = np.arange(1, self.n+1+1)
        self.env_type               = params['env_type'] # one of ['inverted pendulum', 'compound pendulum']
                
        # Cart_Pendulum_Math attributes
        self.constants             = {} # string-symbol-value mappings for params['constants']
        self.variables             = {} # t and \ddot{x_c}
        self.math                  = {} # definition of all string-symbol-function mappings
        self.substitution_mappings = {} # other useful mappings for substitutions
        
        # Cart_Pendulum_Feature_Engineering_Mixin
        self.state_functions       = {} # symbols-function mappings for calculating state-features for feature engineering
        
        # Cart_Pendulum_Simulation_Mixin
        self.state                 = {} # simulation state of a single step; set in self.step(), reset in self.reset()
        
        self.__parameter_sanity_check(params)
        
    def __parameter_sanity_check(self, params):
        constants      = params['constants']
        # Checking type definitions
        assert (self.env_type in ('inverted pendulum', 'compound pendulum')), (f"env_type must be either 'inverted pendulum' or 'compound pendulum'")
            
        # Checking constant definitions            
        list_defined_constants = ['l', 'r', 'm', '\mu', 'I']
        for c_ in list_defined_constants:
            assert (len(constants[c_])  == self.n), (f"{constants[c_]} needs to have n={self.n} entries for {c_}")
        
        if not 'r' in constants: # Parameter correction of r without warning, if r not defined
            constants['r'] = constants['l'] # r will be heavily used for modeling the 'inverted pendulum', so it is always needed.
        if self.env_type == 'inverted pendulum':
            assert (np.prod(constants['r'] == constants['l'])), (f"Every masspoint distance r={constants['r']} should be exactly set to l={constants['l']}, when using type 'inverted pendulum'")
        if self.env_type == 'compound pendulum':
            assert (np.prod(constants['r'] <= constants['l'])), ('every r needs to fullfill r <= l')
        assert (constants['w'] >= constants['w_c']), ('The track width w has to be greater than the cart width.')
      
    # Legacy feature: saving the math core environment        
    # def _save_class_obj_after_init(self):
    #     """ Saves the Cart_Pendulum_Environment class (this class initializes Cart_Pendulum_Environment class by class and then saves the class bundle), if self.save_after_init is set to True.
    #         Saves the hard to calculate math of Cart_Pendulum_Environment, avoiding long loading times of 5 or more minutes. """
    #     with open(self.save_filename + '.pkl', 'wb') as file:
    #         dill.dump(self, file)