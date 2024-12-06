import panel as pn
import pandas as pd
import numpy as np
import dill
import os
import io
import time

from rl_pendulum.ui.home_ui import Home_UI
from rl_pendulum.ui.rl_config_ui import RL_Env_Config_UI
from rl_pendulum.ui.env_viz_ui import Env_Data_Analysis_UI
from rl_pendulum.envs.rl_cart_pendulum.param_manager import RL_Cart_Pendulum_Parameter_Manager
from rl_pendulum.envs.rl_cart_pendulum.env import RL_Cart_Pendulum_Environment
from rl_pendulum.envs.cart_pendulum.param_manager import Cart_Pendulum_Parameter_Manager

class Main_UI():
    """ Main UI holding the every part of the UI, namely the tabsystem with each tab UI and the Main Tamplate.
        This Main_UI has no callbacks directly related to the Main_UI itsself. All inter-UI callbacks are separated into
            the class Main_UI_Callback_Mixin by convention.
    """
    def __init__(self, serve_mode='Browser'):

        # Variables
        self.exp_dirs = None # holds all saving directories for an experiment; returned by Home_UI().prepare_new_experiment_dir()
        self.serve_mode = serve_mode.lower()
        assert (self.serve_mode in ['browser', 'notebook']), ("The specified serve_mode={serve_mode} has to be one of ['browser', 'notebook']!")

        # unpacking each tab's UI component and build their callbacks
        ## unpacking
        self.home_ui = Home_UI()
        self.alert_box = pn.pane.Alert(object="", visible=False, margin=(0,10,10,0), min_height=10) # is empty unless displayed
        self.rl_env_config_ui = RL_Env_Config_UI()
        self.env_data_analysis_ui = Env_Data_Analysis_UI()

        # defining the main tabs
        self.main_tabs =pn.Tabs(
            ('Home',                      self.home_ui.ui), # ui is passed by reference, meaning changes in home_ui affect the 'Home' tab.
            ('Environment Config',        self.rl_env_config_ui.ui),
            ('Environment Data Analysis', self.env_data_analysis_ui.ui),
            ('RL Model Config', []), # architecture + agent
            ('RL Train Config', []), # architecture + agent
            ('Training/Testing', []),
            ('Benchmark Data Analysis', []),
            ('Experiment Data Analysis', []),
            sizing_mode='stretch_both',
            min_width=1000,
            dynamic=False
        )

        # defining the main ui from all components
        main_ui = pn.Column(self.alert_box ,self.main_tabs)
        if self.serve_mode == 'browser':
            self.ui = \
            pn.template.VanillaTemplate(
                header=None,
                sidebar=[],
                main=main_ui,
                modal=[],
                favicon='', # the Tab icon displayed
                logo= '', #'https://panel.holoviz.org/_static/logo_horizontal.png',
                title="RL Pendulum Experiments",
            )
        elif self.serve_mode == 'notebook': # This might look distorted without a template (TODO: test it)
            self.ui = main_ui

class Main_UI_Callback_Mixin:
    """ Main_UI_Callback_Mixin defines any inter-UI callbacks which involve
            components from multiple callback classes by convention.
    """
    def set_callbacks(self):

        ## Callback definition
        self.home_ui.button_start.on_click(self.start_experiment)
        self.home_ui.button_load.on_click(self.load_experiment)
        # rl_config buttons_ui
        self.rl_env_config_ui.button_val_and_continue.on_click(self.validate_rl_conf_and_continue)
        # env_data_analysis_ui buttons, file_input
        self.env_data_analysis_ui.button_initialize_env.on_click(self.initialize_env)
        self.env_data_analysis_ui.button_calculate_sim_data.on_click(self.calulate_simulation_data)
        self.env_data_analysis_ui.sim_parameter_wb.param_to_component['load_dataset'].param.watch(self.load_simulation_dataset, 'value')


    ### callbacks ###
    # (all callbacks that affect the Main_UI's tabs or the Main_UI's AlertBox need to be defined here)
    # For debugging, uncomment the try-except statements for full error messages


    ## Home_UI callbacks ##
    def start_experiment(self, event):
        self.alert_box.visible=False
        exp_base_dir  = self.home_ui.exp_save_dir_input.value
        exp_name = self.home_ui.exp_name_input.value

        try:
            exp_dir = exp_base_dir + '\\' + exp_name
            os.mkdir(exp_dir)
            self.home_ui.exp_save_dir_input.value
            self.home_ui.set_new_exp_name()
            self.exp_dirs = self.home_ui.prepare_new_experiment_dir()

        except Exception as e:
            self.alert_box.object = f"An **Error** occured: {e}"
            self.alert_box.alert_type='primary'
            self.alert_box.visible=True
            print(e)
            # dont continue if something is wrong; but display the problem

        else:
            self.main_tabs.active = 1
            self.rl_env_config_ui.update_exp_file_info(self.exp_dirs['saved data'])
            self.env_data_analysis_ui.update_exp_file_info(self.exp_dirs['saved data'])

    def load_experiment(self, event):
        self.alert_box.visible=False
        loaded_exp_dir = self.home_ui.exp_load_dir.value
        exp_dir  = self.home_ui.exp_save_dir.value
        exp_name = self.home_ui.exp_name_input.value

        try:

            raise NotImplementedError('This button method is not implemented yet (WIP)')
            ###
            # loading experiment code here... TODO
            ###
            os.mkdir(exp_dir + '\\' + exp_name)
            self.home_ui.exp_save_dir.value

        except Exception as e:
            self.alert_box.object = f"An **Error** occured: {e}"
            self.alert_box.alert_type='primary'
            self.alert_box.visible=True
            print(e)
            pass # dont continue if something is wrong; but display the problem

        else:
            self.main_tabs.active = 1

    ## RL_Env_Config_UI() callbacks ##
    def validate_rl_conf_and_continue(self, event):
        self.alert_box.visible=False

        try:
            params = self.rl_env_config_ui.get_param_values()
            rl_reset_params = {**params['rl_init_params'], **params['rl_env_fe_params']}
            manager    = Cart_Pendulum_Parameter_Manager(validate_params=True, **params['math_env_params']) # check 'math_env_params'
            extra_validation_params = {'env_type'  : params['math_env_params']['env_type'],
                                    'n'         : params['math_env_params']['constants']['n'],
                                    'w'         : params['math_env_params']['constants']['w'],
                                    'w_c'       : params['math_env_params']['constants']['w_c']} # pass these to env.validation_reset()
            rl_manager = RL_Cart_Pendulum_Parameter_Manager(validate_params=True, **extra_validation_params)
            params = {**rl_reset_params, **extra_validation_params}
            rl_manager.validation_reset(**params) # check all parameters and do a test reset() call
        except Exception as e:
            self.alert_box.object = f"A **Validation Error** occured: {e}"
            self.alert_box.alert_type='primary'
            self.alert_box.visible=True
            print(e)

        else:
            self.main_tabs.active = 2

    ## Env_Data_Analysis_UI callbacks ##
    def initialize_env(self, event):
        self.alert_box.visible=False

        try:
            t1 = time.perf_counter()
            params = self.rl_env_config_ui.get_param_values() # use the past defined params; there will not be much difference if the user validate these params or not: there will always be an error
            value_ = self.env_data_analysis_ui.sim_parameter_wb.param_to_component['load_env'].value

            if value_ != None:
                # Load the RL_Cart_Pendulum class object from a .pkl file (This is always very fast)
                ## (Security Warning: only use your own .pkl files when possible)
                self.env_data_analysis_ui.rl_sim_env = dill.loads(value_)
                assert (self.env_data_analysis_ui.rl_sim_env.n == params['math_env_params']['constants']['n']), \
                       ('The n parameter from your configuration and your loaded object have to match!')
            else:
                # Initialize the environment from params
                core_params = {**params['math_env_params'],
                               **{'save_after_init' : params['rl_init_params']['save_after_init'],
                                  'save_filename'   : params['rl_init_params']['save_filename'],
                                  'save_dirname'    : params['rl_init_params']['save_dirname']}}
                self.env_data_analysis_ui.rl_sim_env = RL_Cart_Pendulum_Environment(**core_params)

            t2 = time.perf_counter()

        except Exception as e:
            self.alert_box.object = f"An **Error** occured: {e}"
            self.alert_box.alert_type='primary'
            self.alert_box.visible=True
            print(e)

        else:
            whole_minutes = int((t2 - t1)/60)
            whole_seconds_wo_minutes = int((t2 - t1)-whole_minutes*60)
            self.alert_box.object = f"Successfully **Completed** the environment initialization after {str(whole_minutes).zfill(2)}:{str(whole_seconds_wo_minutes).zfill(2)} min." # format like 59:01 min
            self.alert_box.alert_type='success'
            self.alert_box.visible=True # make the alert box invisible after calculating data

    def calulate_simulation_data(self, event):
        self.alert_box.visible=False

        try:
            params = self.rl_env_config_ui.get_param_values() # use the past defined params; there will not be much difference if the user validate these params or not: there will always be an error
            reset_params = {**params['rl_init_params'], **params['rl_env_fe_params']}
            self.env_data_analysis_ui.rl_sim_env.reset(**reset_params)
            T = self.env_data_analysis_ui.rl_sim_env.T
            dt = params['rl_init_params']['dt']
            t_step = np.arange(dt,T+dt,dt) # getting stepsize [0, T] as closed interval in steps of dt; step for t=0 already done with env.reset!
            control_func_str = self.env_data_analysis_ui.sim_parameter_wb.param_to_component['control_func'].value
            control_func     = self.env_data_analysis_ui.sim_parameter_wb.str_to_control_func[control_func_str]

            t1 = time.perf_counter()

            for t_ in t_step:
                a_t = control_func(t_)
                state_, r_ = self.env_data_analysis_ui.rl_sim_env.step(a_t=a_t)

            t2 = time.perf_counter()

            info = self.env_data_analysis_ui.rl_sim_env.get_step_time_info(sample_steps=10, agent_func=None)
        except Exception as e:
            self.alert_box.object = f"An **Error** occured: {e}"
            self.alert_box.alert_type='primary'
            self.alert_box.visible=True
            print(e)

        else:
            whole_minutes = int((t2 - t1)/60)
            whole_seconds_wo_minutes = int((t2 - t1)-whole_minutes*60)
            self.alert_box.object = f"Successfully **Calculated** the simulation data after {str(whole_minutes).zfill(2)}:{str(whole_seconds_wo_minutes).zfill(2)} min." # format like 59:01 min
            self.alert_box.alert_type='success'
            self.alert_box.visible=True # make the alert box invisible after calculating data

            self.env_data_analysis_ui.update_data(df = self.env_data_analysis_ui.rl_sim_env.get_buffer_as_df(), info=info)

    def load_simulation_dataset(self, event):
        try:
            t1 = time.perf_counter()

            byte_str = self.env_data_analysis_ui.sim_parameter_wb.param_to_component['load_dataset'].value
            file_object = io.BytesIO(byte_str)
            df = pd.read_csv(file_object)
            df.set_index(df.columns[0], drop=True, inplace=True) # The first column is always the index; this code works for all names
            if df.index.name == 'Unnamed: 0':
                df.index.name = 'Index'

            t2 = time.perf_counter()

        except Exception as e:
            self.alert_box.object = f"An **Error** occured: {e}"
            self.alert_box.alert_type='primary'
            self.alert_box.visible=True
            print(e)

        else:
            whole_minutes = int((t2 - t1)/60)
            whole_seconds = int((t2 - t1)-whole_minutes*60)
            self.alert_box.object = f"Successfully **Loaded** the simulation data after {str(whole_minutes).zfill(2)}:{str(whole_seconds).zfill(2)} min." # format like 59:01 min
            self.alert_box.alert_type='success'
            self.alert_box.visible=True # make the alert box invisible after calculating data

            self.env_data_analysis_ui.update_data(df=df, info=None)
