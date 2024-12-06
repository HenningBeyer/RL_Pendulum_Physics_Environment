from rl_pendulum.envs.cart_pendulum.env import Cart_Pendulum_Environment
from rl_pendulum.envs.rl_cart_pendulum.core import RL_Cart_Pendulum_Environment_Core
from rl_pendulum.envs.rl_cart_pendulum.vec_core import RL_Cart_Pendulum_Environment_Vec_Core
from rl_pendulum.envs.rl_cart_pendulum.param_manager import RL_Cart_Pendulum_Parameter_Manager
from rl_pendulum.envs.rl_cart_pendulum.fe import RL_Cart_Pendulum_Feature_Engineering
from rl_pendulum.envs.rl_cart_pendulum.bench import RL_Cart_Pendulum_Benchmark_Mixin

class RL_Cart_Pendulum_Environment(Cart_Pendulum_Environment,
                                   RL_Cart_Pendulum_Environment_Core,
                                   RL_Cart_Pendulum_Environment_Vec_Core,
                                   RL_Cart_Pendulum_Parameter_Manager,
                                   RL_Cart_Pendulum_Feature_Engineering,
                                   RL_Cart_Pendulum_Benchmark_Mixin,
                                   ):
    """ This is the full RL_Cart_Pendulum_Environment class. It only needs the parameters for Cart_Pendulum_Environment to be initialized.
        But this class will need much more parameters to call self.reset to prepare the class for the first episode.
    """

    def __init__(self, **params):
        temp_ = params['save_after_init'] if 'save_after_init' in params.keys() else False
        params['save_after_init'] = False # prevent Cart_Pendulum_Environment from automatically saving upon initialization
        Cart_Pendulum_Environment.__init__(self, **params)
        params['save_after_init'] = temp_
        RL_Cart_Pendulum_Environment_Core.__init__(self, **params) # RL_Cart_Pendulum_Environment_Core needs Cart_Pendulum_Math initialized to do env.reset; __init__() does nothing here
        RL_Cart_Pendulum_Parameter_Manager.__init__(self, **params) # The manager does some minor overwrites
        if self.save_after_init:
            self._save_class_obj_after_init()