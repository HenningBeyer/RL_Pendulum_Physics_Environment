import dill
import numpy as np

from rl_pendulum.envs.rl_cart_pendulum.core import RL_Cart_Pendulum_Environment_Core
from rl_pendulum.envs.rl_cart_pendulum.fe import RL_Cart_Pendulum_Feature_Engineering

from rl_pendulum.utils.feature_engineering import Feature_Engineering_Param_Provider, Feature_Engineering_Mapping_Provider
from rl_pendulum.envs.rl_cart_pendulum.fe_buffer import FE_Replay_Buffer
from rl_pendulum.envs.rl_cart_pendulum.episode_initializer import Episode_Initializer, Random_Episode_Initializer
import os

class RL_Cart_Pendulum_Parameter_Manager():
    """ Class which unpacks parameters and checks them for correct definition.
        It also pre-defines variables for clarity.

        class that pre-defines every possible attribute in advance for clarity and has no effect on other classes.
        This is the first class instantiated when instantiating RL_Cart_Pendulum_Environment class.

        Note: This RL_Cart_Pendulum_Parameter_Manager class is also often called by other parts of the bundle class RL_Cart_Pendulum_Environment
            --> (The Parameter_Manager class should do all work with parameters; the RL_Environment_Core class should give all basic RL functions; ...)
            --> All parameter setting and checking is moved to this class
            --> This helps to categorize each class's functionality better.

        This class is also called by the UI to do Parameter sanity checks
    """
    def __init__(self, **params):
        """ These are parameters present in Cart_Pendulum_Environment which can be overwritten via the RL_Cart_Pendulum_Parameter_Manager. """

        if 'save_after_init' in params.keys():
            self.save_after_init = params['save_after_init']
        else:
            self.save_after_init = False

        if 'save_filename' in params.keys():
            self.save_filename = params['save_filename']
        elif hasattr(self, 'n') and hasattr(self, 'env_type'):
            self.save_filename = self.save_filename = f"RL_env_n{self.n}_{self.env_type.replace(' ', '_')}_obj" # the name should suggest that it is a general object; as all behaviour can be altered by self.reset (except the math constants)
        else:
            self.save_filename = None # This should only happen for a parameter validation

        if 'save_dirname' in params.keys():
            self.save_dirname = params['save_dirname']
        else:
            self.save_dirname = os.getcwd()

    def validation_reset(self, **params):
        """ This function calls self._env_reset_parameter_sanity_check(**params) to validate all parameter definitions.
            All other errors should be coding errors.
        """
        # Constants of the math environment which need to be defined, in order to run validation_reset()
        self.n         = params['n']
        self.constants = {'str_to_val' : {'w_c' : params['w_c'], 'w' : params['w']}}
        self.n_range   = np.arange(1, self.n+1)
        self.env_type  = params['env_type']
        self.bump_x     = (self.constants['str_to_val']['w']-self.constants['str_to_val']['w_c'])/2


        self.reset_episode_params(**params)
        self.set_feature_engineering_params(**params)
        self._env_reset_parameter_sanity_check()

        # These functions would need RL_Cart_Pendulum_Feature_Engineering and RL_Cart_Pendulum_Environment_Core to be initialized together
        ## This is possbile by doing Dummy_Class(RL_Cart_Pendulum_Environment_Core, RL_Cart_Pendulum_Feature_Engineering)
        ### But this is very bad programming, and might not work for slight changes
        ### Overall, just the _env_reset_parameter_sanity_check should be needed, when the code runs without errors
        #self.build_feature_engineering() # dependend on the two functions above, needs RL_Cart_Pendulum_Feature_Engineering to be inherited from
        #self.define_episode_variables()  # dependend on the two functions above
        #self.reset_step(**params)


    def set_feature_engineering_params(self, **params):
        """ These parameters must only be reset once after initializing the math environment.
            But these parameters are still reset for every env.reset() out of convenience (no extra function call).
        """

        # general feature engineering parameters
        self.do_feature_engineering      = params['do_feature_engineering']

        # classic feature engineering
        self.do_classic_engineering      = params['CFE']['do_classic_engineering']
        self.feature_groups              = params['CFE']['feature_groups']

        # derivative feature engineering
        self.do_derivative_engineering   = params['DFE']['do_derivative_engineering']
        self.dfe_feature_groups          = params['DFE']['dfe_feature_groups']
        self.dfe_included_base_features  = params['DFE']['dfe_included_base_features']
        self.max_derivative_order        = params['DFE']['max_derivative_order']

        # time series feature engineering
        self.do_ts_feature_engineering   = params['TSFE']['do_ts_feature_engineering']
        self.do_difference_tsfe          = params['TSFE']['do_difference_tsfe']
        self.do_overlapping_win_tsfe     = params['TSFE']['do_overlapping_win_tsfe']
        self.do_non_overlapping_win_tsfe = params['TSFE']['do_non_overlapping_win_tsfe']
        self.tsfe_features               = params['TSFE']['tsfe_features']
        self.tsfe_lookback_mode          = params['TSFE']['tsfe_lookback_mode']

        self.overlapping_fwin_aggfuncs          = params['TSFE']['overlapping_fwin_aggfuncs']
        self.overlapping_now_fwin_aggfuncs      = params['TSFE']['overlapping_now_fwin_aggfuncs']
        self.overlapping_ffwin_fwin_aggfuncs    = params['TSFE']['overlapping_ffwin_fwin_aggfuncs']
        self.non_overlapping_fwin_aggfuncs      = params['TSFE']['non_overlapping_fwin_aggfuncs']
        self.non_overlapping_now_fwin_aggfuncs  = params['TSFE']['non_overlapping_now_fwin_aggfuncs']
        self.non_overlapping_ffwin_fwin_aggfuncs = params['TSFE']['non_overlapping_ffwin_fwin_aggfuncs']

    def build_feature_engineering(self):
        """ To call this RL_Cart_Pendulum_Feature_Engineering needs to be inherited from. """
        # set names for state columns
        x_c_cols     = ['$$x_c$$', r'$$\dot{x_c}$$', r'$$\ddot{x_c}$$']
        reward_cols  = ['$$r$$']
        theta_cols   = [fr'$$\theta_{{{i}}}$$' for i in self.n_range]
        dtheta_cols  = [fr'$$\dot{{\theta_{i}}}$$' for i in self.n_range]
        ddtheta_cols = [fr'$$\ddot{{\theta_{i}}}$$' for i in self.n_range]

        self.base_col_names = x_c_cols + reward_cols + theta_cols + dtheta_cols + ddtheta_cols # This sets the column order for the later code
        self.reset_feature_engineering() # method of RL_Cart_Pendulum_Feature_Engineering
        self.s2i = self.s2i # self.s2i is set by self.reset_feature_engineering(); explicit writing as of importance

    def reset_episode_params(self, **params):    # called RL_Cart_Pendulum_Environment_Core.reset()
        """ Function for setting all attributes related to an episode. """
        # environment params per episode
        self.dt                          = params['dt'] # step_time
        self.max_cart_acceleration       = params['max_cart_acceleration']
        self.equilibrium_mode            = params['equilibrium_mode']

        self.episode_initializer_type    = params['episode_initializer_type']
        if self.episode_initializer_type == 'Episode_Initializer':
            self.episode_initializer     = Episode_Initializer(**params)
        elif self.episode_initializer_type == 'Random_Episode_Initializer':
            self.episode_initializer     = Random_Episode_Initializer(**{**params, **{'n' : self.n}})
        else:
            raise ValueError(F"The specified episode_initializer_type {self.episode_initializer_type} does not refer to a supported option.")

        init_params = self.episode_initializer.sample()
        self.T                           = init_params['T'] # episode time
        self.initial_cart_pos            = init_params['initial_cart_pos']
        self.initial_cart_velocity       = init_params['initial_cart_velocity']
        self.initial_angles              = init_params['initial_angles']
        self.initial_velocities          = init_params['initial_velocities']

    def define_episode_variables(self):
        """ These get an extra function as they depend of feature engineering attributes, i.e. need to be defined after feature engineering
        """
        self.t_index    = np.arange(0,self.T+self.dt,self.dt) # needing self.dt steps, needing steps of 0 and  step T' >= T to be included; T' == T only when T % dt == 0
        self.max_steps  = len(self.t_index)
        self.t          = 0.0
        self.step_num   = 0
        self.feat_count = len(self.s2i.keys())        # defined in env.reset_step() after feature engineering
        self.bump_x     = (self.constants['str_to_val']['w']-self.constants['str_to_val']['w_c'])/2 # needs math constants
        self.did_bump   = False
        self.buffer     = FE_Replay_Buffer(max_size=self.max_steps, num_feats=self.feat_count) # The buffer is a np.array as of greater performance over torch.tensor inside feature engineering



    def _env_reset_parameter_sanity_check(self): # called after self.reset()
        """ There is no module that does very custom parameter checking, so it is implemented here. """
        param_provider  = Feature_Engineering_Param_Provider()

        # Common parameter check
        assert ((type(self.equilibrium_mode) == type(str())) and \
                (len(self.equilibrium_mode) == self.n) and \
                (self.equilibrium_mode.replace('0', '').replace('1', '') == '')), (f"The equilibrium_mode {self.equilibrium_mode} is specified wrong in length or symbols.")  # mode like '1111' for a standing 4-rod pendulum
        assert ((len(self.initial_angles) == self.n) and (len(self.initial_velocities) == self.n)), (f"initial_velocities and initial_angles need to be of length {self.n}")
        assert (self.dt <= 0.05), ('There should only be sampling rates dt <= 50 ms for precision.')
        assert (np.abs(self.initial_cart_pos) <= self.bump_x), (f"The initial_cart_pos {self.initial_cart_pos} lies outside the track bounds: bump_x = (track_width-cart_width)/2 = {bump_x}")

        #if (self.do_derivative_engineering == True) or (self.do_ts_feature_engineering == True):
            #assert (self.do_classic_engineering == True), ("Classic feature engineering (CFE) has to be enabled when doing CFE or TSFE.")
        assert ( (type(self.max_derivative_order) == type(int())) and (self.max_derivative_order in [0,1,2]) )
        assert (self.env_type == 'compound pendulum'), \
               ("The only option of self.env_type is 'compound pendulum', as of the more general feature engineering. "+\
                "Any behaviour of the 'inverted pendulum' can be achieved by setting all r to l")
        assert (self.max_cart_acceleration > 0), ('Do not pick negative max accelerations.')

        # Feature engineering check
        ## Classical feature engineering
        all_feature_groups = param_provider.get_all_feature_groups()
        for fg_ in self.feature_groups:
            assert (fg_ in all_feature_groups), \
                   (f"The feature group {fg_} is misspelled or not part of the original feature group set. all_feature_groups: {all_feature_groups}")

        ## Derivative feature engineering
        allowed_feats = [r'$$\ddot{x_c}$$', '$$r$$']
        for feat_ in self.dfe_included_base_features:
            assert (feat_ in [r'$$\ddot{x_c}$$', '$$r$$']), (fr"The only options for self.dfe_included_base_features are {allowed_feats}.")
        all_deriv_featgroups = param_provider.get_all_feature_groups()
        for fg_ in self.dfe_feature_groups:
            assert (fg_ in all_deriv_featgroups), \
                   (f"The feature group {fg_} in dfe_feature_groups {dfe_feature_groups} is misspelled or not part of the original feature set: {all_deriv_featgroups}")

        ## Time series feature engineering
        all_tsfe_feats = param_provider.get_all_tsfe_features(self.n)
        for feat_ in self.tsfe_features:
            assert (feat_ in all_tsfe_feats), \
                   (f"The feature {feat_} of 'tsfe_features' is misspelled or not part of the original feature set. all_tsfe_feats: {all_tsfe_feats}")
        assert (self.tsfe_lookback_mode in ['long', 'medium', 'short'])

        fwin_func_groups     = ['Mean', 'Min', 'Max', 'Std']
        now_fwin_func_groups = ['Now-Mean', 'Now-Min', 'Now-Max']
        firstfwin_fwin_func_groups = ['Mean-Mean', 'Mean-Min', 'Mean-Max',
                                      'Min-Mean',  'Min-Min',  'Min-Max',
                                      'Max-Mean',  'Max-Min',  'Max-Max',
                                      'Std-Std']

        for fgroup_ in self.overlapping_fwin_aggfuncs:
            assert (fgroup_ in fwin_func_groups), (f"All elements of self.overlapping_fwin_aggfuncs {self.overlapping_fwin_aggfuncs} must be in {fwin_func_groups}.")
        for fgroup_ in self.overlapping_now_fwin_aggfuncs:
            assert (fgroup_ in now_fwin_func_groups), (f"All elements of self.overlapping_now_fwin_aggfuncs {self.overlapping_now_fwin_aggfuncs} must be in {now_fwin_func_groups}.")
        for fgroup_ in self.overlapping_ffwin_fwin_aggfuncs:
            assert (fgroup_ in firstfwin_fwin_func_groups), (f"All elements of self.overlapping_ffwin_fwin_aggfuncs {self.overlapping_ffwin_fwin_aggfuncs} must be in {firstfwin_fwin_func_groups}.")

        for fgroup_ in self.non_overlapping_fwin_aggfuncs:
            assert (fgroup_ in fwin_func_groups), (f"All elements of self.non_overlapping_fwin_aggfuncs {self.non_overlapping_fwin_aggfuncs} must be in {fwin_func_groups}.")
        for fgroup_ in self.non_overlapping_now_fwin_aggfuncs:
            assert (fgroup_ in now_fwin_func_groups), (f"All elements of self.non_overlapping_now_fwin_aggfuncs {self.non_overlapping_now_fwin_aggfuncs} must be in {now_fwin_func_groups}.")
        for fgroup_ in self.non_overlapping_ffwin_fwin_aggfuncs:
            assert (fgroup_ in firstfwin_fwin_func_groups), (f"All elements of self.non_overlapping_ffwin_fwin_aggfuncs {self.non_overlapping_ffwin_fwin_aggfuncs} must be in {firstfwin_fwin_func_groups}.")

        # CFE, DFE and TSFE
        for fg_ in self.dfe_feature_groups:
            assert (fg_ in self.feature_groups), (f"Derivative feature engineering is done with the feature group {fg_} which is not in self.feature_groups {self.feature_groups} of classical feature engineering." + \
                                                 f" Add this feature group manually.")

        mapping_provider = Feature_Engineering_Mapping_Provider()
        params_provider = Feature_Engineering_Param_Provider()
        feat2fgroup = mapping_provider.get_feature_to_feature_group_mapping(self.n)
        base_feats  = params_provider.get_base_features(self.n)
        for feat_ in self.tsfe_features:
            if feat_ in base_feats: # these are not included in feat2fgroup
                continue
            feat_group_ =  feat2fgroup[feat_] # get the corresponding feature group
            assert (feat_group_ in self.feature_groups), (f"Time series feature engineering is done with a feature {feat_} which belongs to the feature group {feat_group_} of classical feature engineering." + \
                                                         f" But {feat_group_} is not part of self.feature_groups: {self.feature_groups}. Consider to add it.")
                                                            # It is better to add each feature_group by hand than instead of automaitcally handling this. This avoids mistakes in research.

        # Check FE param redundancy for boolean selections
        ## wrong selection cause some errors too, and give wrong experiment analysis
        if self.do_classic_engineering or self.do_derivative_engineering or self.do_ts_feature_engineering:
            assert (self.do_feature_engineering == True), (f'Set self.do_feature_engineering = True, when using any sub-process of feature engineering.')
        if self.do_derivative_engineering and len(self.dfe_feature_groups) >= 1:
            assert (self.do_classic_engineering), ('You need to set self.do_classic_engineering = True, whenever doing DFE with features of CFE')
        if self.do_difference_tsfe or self.do_overlapping_win_tsfe or self.do_non_overlapping_win_tsfe:
            assert (self.do_ts_feature_engineering == True), (f'Set self.do_ts_feature_engineering = True, when using any sub-process of time series feature engineering.')

        # if self.do_feature_engineering == True:
        #     assert (len(self.feature_groups) >= 1), ('You specified to do feature engineering, but selected no feature group! This is selection is a redundant parameter.')

        if self.do_classic_engineering == True:
            assert (len(self.feature_groups) >= 1), ('You specified to do classical feature engineering, but selected no feature group! This is selection is a redundant parameter.')
        else:
            assert (len(self.feature_groups) == 0), ('You specified to do no classical feature engineering, but selected a feature group! This is selection is a redundant parameter.')

        if self.do_derivative_engineering == True:
            assert (len(self.dfe_feature_groups) >= 1 or len(self.dfe_included_base_features) >= 1),\
                   ('You specified to do classical feature engineering, but selected no base feature or feature group for it! This is selection is redundant.')
        else:
            assert (len(self.dfe_feature_groups) == 0 and len(self.dfe_included_base_features) == 0),\
                   ('You specified to do no classical feature engineering, but selected a feature or feature group for it! This is selection is redundant.')

        if self.do_ts_feature_engineering == True:
            assert (len(self.tsfe_features) >= 1), ('You set self.do_ts_feature_engineering = True, but you did not select any TSFE feature for it! This is a redundant selection.')
            assert (self.do_difference_tsfe or self.do_non_overlapping_win_tsfe or self.do_overlapping_win_tsfe), ('When specifying to do TSFE with do_ts_feature_engineering, at least one of [do_difference_tsfe, do_non_overlapping_win_tsfe, do_overlapping_win_tsfe] must be True. Else, there is a parameter redundancy!')
        else:
            assert (len(self.tsfe_features) == 0), ('You set self.do_ts_feature_engineering = False, but you did select features for TSFE! This is a redundant selection.')
            assert not (self.do_difference_tsfe or self.do_non_overlapping_win_tsfe or self.do_overlapping_win_tsfe), ('You specified one of [do_difference_tsfe, do_non_overlapping_win_tsfe, do_overlapping_win_tsfe] to be True, but you do no TSFE as do_ts_feature_engineering is set to False.This is a parameter redundancy!')

        if self.do_difference_tsfe == True:
            pass

        if self.do_overlapping_win_tsfe == True:
            assert (len(self.overlapping_fwin_aggfuncs) >= 1 or len(self.overlapping_now_fwin_aggfuncs) >= 1 or len(self.overlapping_ffwin_fwin_aggfuncs) >= 1),\
                   ('self.do_overlapping_win_tsfe was set to True, but no aggregation function was selected for feature calculation! This is a redundant selection. ')
        else:
            assert (len(self.overlapping_fwin_aggfuncs) == 0 and len(self.overlapping_now_fwin_aggfuncs) == 0 and len(self.overlapping_ffwin_fwin_aggfuncs) == 0),\
                   ('self.do_overlapping_win_tsfe was set to False, but at least one aggregation function was still selected for feature calculation! This is a redundant selection. ')

        if self.do_non_overlapping_win_tsfe == True:
            assert (len(self.non_overlapping_fwin_aggfuncs) >= 1 or len(self.non_overlapping_now_fwin_aggfuncs) >= 1 or len(self.non_overlapping_ffwin_fwin_aggfuncs) >= 1),\
                   ('self.non_overlapping_ffwin_fwin_aggfuncs was set to True, but no aggregation function was selected for feature calculation! This is a redundant selection. ')
        else:
            assert (len(self.non_overlapping_fwin_aggfuncs) == 0 and len(self.non_overlapping_now_fwin_aggfuncs) == 0 and len(self.non_overlapping_ffwin_fwin_aggfuncs) == 0),\
                   ('self.non_overlapping_ffwin_fwin_aggfuncs was set to False, but at least one aggregation function was still selected for feature calculation! This is a redundant selection. ')

    def _save_class_obj_after_init(self): # This is an optional override; this is the same method as in Cart_Pendulum_Parameter_Manager; it could be redefined here for RL_Cart_Pendulum_Environment
        """ Saves the Cart_Pendulum_Environment class object after called internally, if self.save_after_init is set to True.
            Saves the hard to calculate math of Cart_Pendulum_Environment, avoiding long loading times of 5 or more minutes. """
        with open(self.save_dirname + '\\' + self.save_filename + '.pkl', 'wb') as file:
            dill.dump(self, file)