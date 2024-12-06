from rl_pendulum.envs.cart_pendulum.param_manager import Cart_Pendulum_Parameter_Manager
from rl_pendulum.envs.cart_pendulum.math import Cart_Pendulum_Math
from rl_pendulum.envs.cart_pendulum.sim_mixin import Cart_Pendulum_Simulation_Mixin
from rl_pendulum.envs.cart_pendulum.viz_mixin import Cart_Pendulum_Visualization_Mixin

class Cart_Pendulum_Environment(Cart_Pendulum_Parameter_Manager,        # Parent class that is responsible to store all potential parameters for the Pendulum environment and its mixin extensions.
                                Cart_Pendulum_Math,                     # Parent class of Cart_Pendulum_Environment, which needs to be initialized to provide self.math
                                Cart_Pendulum_Simulation_Mixin,          # Mixin class; uses self.math in its methods; provides simulation functions
                                Cart_Pendulum_Visualization_Mixin        # Mixin class that just adds methods; uses self.math in its methods; provides animation, plotting, and math summarization
                               ):
    """ This is the full pendulum class with Cart_Pendulum_Parameter_Manager, Cart_Pendulum_Math and its optional mixin classes """
    def __init__(self, **params):
        """ Initializes Cart_Pendulum_Parameter_Manager and Cart_Pendulum_Math (parent classes) 
            Mixin classes are not initialized; they have by definition no __init__().
        """
        Cart_Pendulum_Parameter_Manager.__init__(self, **params)
        Cart_Pendulum_Math.__init__(self, **params['constants'])
        # All the other mixins provide all the functionalities of Cart_Pendulum_Environment by using self.math
        # Mixin class function get also their own parameters, but only when calling them by their methods
                
        # Legacy feature: saving the math core environment 
        # if self.save_after_init:
        #     self._save_class_obj_after_init()

        
        
