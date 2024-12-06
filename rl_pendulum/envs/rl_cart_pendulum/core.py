import numpy as np
import pandas as pd
import time
from scipy.integrate import solve_ivp

class RL_Cart_Pendulum_Environment_Core():
    """ RL_Cart_Pendulum_Environment_Core is the core class of RL_Cart_Pendulum_Environment.
        - This class should define the step(), reset(), and get_reward() functions to enable the deployment of RL systems.
        - This class depends on Cart_Pendulum_Environment to be set for the step function
    """

    def __init__(self, **params):
        pass # Nothing to be done here; call env.reset to set the class parameters

    def reset(self, **params):
        """ This function manages the complete reset() process of the environment. """
        self.bump_x     = (self.constants['str_to_val']['w']-self.constants['str_to_val']['w_c'])/2 # defining bump_x for self._env_reset_parameter_sanity_check()

        self.reset_episode_params(**params)
        self.set_feature_engineering_params(**params)
        self._env_reset_parameter_sanity_check()
        self.build_feature_engineering() # depends on the two functions above, needs RL_Cart_Pendulum_Feature_Engineering to be inherited from
        self.define_episode_variables()  # depends on the two functions above
        self.reset_step(**params)


    def reset_step(self, **params):
        """ Sets the first step's data """

        # initialize the initial state for the agent
        reward         = self.get_reward(np.array(self.initial_angles))
        action_t0      = 0.0
        ddthetas       = self.n*[0] # cant be calculated easily
        raw_state = np.concatenate(([self.initial_cart_pos], [self.initial_cart_velocity], [action_t0], [reward],
                                     self.initial_angles, self.initial_velocities, ddthetas)) # column order defined by self.base_col_names! Do not alter

        # Doing first-step feature engineering
        if self.do_feature_engineering==True:
            raw_state  = np.pad(raw_state, pad_width=(0, self.feat_count-len(raw_state)), mode='constant', constant_values=(np.nan)) # replace features not calculated in feature engineering withth np.nan
            self.buffer.update(np.array([raw_state])) # this state is completed with self.run_fe_pipeline()
            buffer_idx = self.buffer.get_last_state_idx()-1
            self.run_fe_pipeline(self.buffer.buffer, buffer_idx) # this updates the last state array inside buffer.buffer by reference
                                                                          # passes the entire replay buffer np.array by reference --> no speed/memory issues
        else:
            self.buffer.update(np.array([raw_state])) # 2D array as input

        self.t += self.dt
        self.state = self.buffer.sample_last_state()
        return self.state, reward

    def reset_vec_step(self):
        pass

    def convert_net_2pi_angles_to_pi_angles(self, angles : np.array):
        """ Convert joint angles \theta \in [-inf; inf] (uses net [0; +-2pi] rotation) --> to [0, 2pi] --> to [pi, -pi].
            (The zero angle is always upwards, positive in the left direction) """

        # Converting net [0; +-2pi] angles (means (-inf; +inf) angles with two-directional rotation)
        #  to [0, 2pi] angles (one directional rotation)
        freq   = np.pi*2
        n      = angles // (freq)
        angles = angles - n*freq

        # Converting [0, 2pi] angles to [pi, -pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        return angles


    def get_reward(self, net_thetas : np.array):
        """ There can be 2^n different equilibria for a n-rod pendulum; each mode is represented using a binary number, where each digit stands for standing (1) or hanging down (0)
            Examples: eq_mode=16; eq_mode-1 --> 1111 (all standing), 0 --> 0000 (all hanging), 2 --> 0010 (only rod 2 is standing, rest hanging).

            Gives possibility to stabilize at any wished equilibrium via reward feedback.
        """
        thetas = self.convert_net_2pi_angles_to_pi_angles(net_thetas) # ranges of [-pi, pi]
        thetas = np.abs(thetas) # Converting to [pi, -pi] angles to [0, pi]
        rod_rewards = [np.cos(thetas[i-1])
                       if self.equilibrium_mode[i-1] == '1'         # standing
                       else np.cos( np.abs(thetas[i-1] - np.pi) )   # hanging (equilibrium_mode[i-1] == '0')
                       for i in self.n_range]
        reward = np.mean(rod_rewards) # ranges of [-1, 1]
        if self.did_bump:
            reward += -1
        return reward


    def step(self, a_t):
        """ Calulates the next step given an input acceleration a.
            If doing no feature engineering and setting self.do_feature_engineering from True to False, self.step takes ~3ms less.
                step times without feature engineering for different n: {1 : 2-3ms, 2 : 2-3ms, 3 : 4-5ms, 4 : 15-16ms, 5 : 95-115ms}
        """

        # unpacking of the last state
        s,v,a          = self.state[0:3] # cart position, cart velocity, cart acceleration
        initial_values = self.state[4:4+2*self.n] # theta and dtheta values

        # bumping penalty
        bumped = (np.abs(s) >= self.bump_x)
        if bumped:
            self.did_bump = True
        else:
            self.did_bump = False

        # step calculation
        a_t = np.sign(a_t)*np.min( (np.abs(a_t), np.abs(self.max_cart_acceleration)) )
        v_t = v + a_t*self.dt
        s_t = s + v_t*self.dt + a_t*self.dt*self.dt/2

        robotic_equation = self.math['str_to_sol_func']['ddq_f']
        angular_data  = solve_ivp(lambda t_,x_:
                                  robotic_equation(*x_, a_t, *self.ddqf_constants['sym_to_val'].values()),
                                  t_span=[0.0,self.dt], y0=initial_values, method='RK45', t_eval=[0,self.dt], rtol=1e-6)
        # angular_data['y'] --> [[ 3.14159265e+00,  3.14159265e+00], # theta_1
        #                        [ 3.14159265e+00,  3.14159265e+00], # theta_2
        #                        [ 0.00000000e+00,  4.32886225e-18], # \dot{theta_1}
        #                        [ 0.00000000e+00, -3.61154422e-19]] # \dot{theta_2} for the first step with n=2
        data = np.array(angular_data['y'])
        thetas_t       = data[       : self.n, 1]
        dthetas_t      = data[self.n :       , 1]
        dthetas_t_past = data[self.n :       , 0]
        ddthetas_t = (dthetas_t - dthetas_t_past)/self.dt # works fine like an actual derivative function

        # reward
        reward = self.get_reward(thetas_t)

        # feature engineering / state_df assignment
        raw_state = np.concatenate((np.array([s_t, v_t, a_t, reward]), thetas_t, dthetas_t, ddthetas_t)) # column order defined by self.base_col_names! Do not alter
        if self.do_feature_engineering==True:
            raw_state = np.pad(raw_state, pad_width=(0, self.feat_count-len(raw_state)), mode='constant', constant_values=(np.nan)) # replace features not calculated in feature engineering withth np.nan
            self.buffer.update(np.array([raw_state]))
            buffer_idx = self.buffer.get_last_state_idx() - 1
            self.run_fe_pipeline(self.buffer.buffer, buffer_idx) # this updates the last state array inside buffer.buffer by reference
        else:
            self.buffer.update(np.array([raw_state])) # 2D array as input

        self.t += self.dt
        self.step_num += 1
        self.state = self.buffer.sample_last_state()
        return self.state, reward

    def get_step_time_info(self, sample_steps=10, agent_func=None):
        """ This function is for providing step time stats within a few iterations """
        # env step
        sample_state = self.buffer.sample_last_state() # these are occasionally dummy values
        self.buffer.buffer_is_static = True # prevent the buffer from updating
        self.buffer.p = self.buffer.max_size # set the buffer index to the last index to fully calculate TSFE features
        t_env   = 0.0
        t_fe    = 0.0
        t_agent = 0.0
        for i in range(sample_steps):

            # env step time (has to include feature engineering to not have to reset all FE constants and the buffer)
            t1 = time.perf_counter()
            self.step(a_t=0.1)
            t = time.perf_counter() - t1
            t_env += t

            # feature engineering step time (full feature engineering)
            if self.do_feature_engineering:
                t1 = time.perf_counter()
                self.run_fe_pipeline(self.buffer.buffer, buffer_idx=self.buffer.max_size-1) # This runs fe with the full buffer; TSFE runs with a complete time series history
                t = time.perf_counter() - t1
                t_fe += t

            # agent step time
            if agent_func != None:
                t1 = time.perf_counter()
                agent_func(sample_state)
                t = time.perf_counter() - t1
                t_agent += t

        # taking the mean
        t_env   /= sample_steps
        t_fe    /= sample_steps
        t_agent /= sample_steps

        # removing FE time from t_env, if FE was done
        if self.do_feature_engineering:
            t_env -= t_fe

        # allow further buffer updating
        self.buffer.buffer_is_static = False
        self.buffer.p = 0

        return {'env' : t_env, 'fe' : t_fe, 'agent' : t_agent}

    def get_buffer_as_df(self):
        """ Returns the buffer as indized DataFrame (for data analysis) """
        # if as_latex == False:
        #     columns = [str_.replace('$$', '') for str_ in self.s2i.keys()] # panel does currently not support r'$$\frac{\partial r}{\partial t}$$' latex strings as DataFrame header
        # else:
        columns = self.s2i.keys()
        df = pd.DataFrame(self.buffer.buffer, columns=columns, index=self.t_index)
        df.index.name = '$$t$$'
        return df


    # TODO: reward norms!!!