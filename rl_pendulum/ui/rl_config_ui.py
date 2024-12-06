import panel as pn
import numpy as np
import os

from rl_pendulum.utils.feature_engineering import Feature_Engineering_Param_Provider
from rl_pendulum.ui.styling import wb_h2_header_margins, divider_margins, latex_margins, \
                                   slider_bar_color, wb_h1_header_margins,\
                                   wb_h1_tooltip_margins, main_wb_margins, sub_wb_margins,\
                                   button_h1_margins, sub_tab_margins,config_sub_wb_margins

class RL_Env_Config_UI():
    def __init__(self):
        
        # unpacking each widgetbox's UI component and build their callbacks
        ## unpacking
        self.math_env_params_ui    = Math_Env_Params_UI()
        self.rl_env_init_params_ui = RL_Env_Init_Params_UI()
        self.rl_env_fe_params_ui   = RL_Env_FE_Params_UI()
        
        ## callback definition
        #home_ui.button_start.on_click(self.start_experiment)
        #rl_env_config_ui.button_coninue.on_click(self.continue_from_rl_env_config)
        
        self.button_val_and_continue = pn.widgets.Button(name="Validate and Continue", margin=button_h1_margins) 
        self.math_env_params_ui.param_to_component['n'].param.watch(self.update_on_n_change, 'value')
        self.math_env_params_ui.param_to_component['w'].param.watch(self.update_cart_pos_interval, 'value')
        self.math_env_params_ui.param_to_component['w_c'].param.watch(self.update_cart_pos_interval, 'value')
        
        self.desc_mapping = {
            'general_info' :  "- Define your environment for your RL model here. \n" +\
                              "- Press 'Validate and Continue' to properly proceed after finishing.",
        }
        self.ui = pn.WidgetBox(
            pn.Row(
                pn.pane.Markdown('# RL Environment Params', margin=wb_h1_header_margins),
                pn.widgets.TooltipIcon(value=self.desc_mapping['general_info'], margin=wb_h1_tooltip_margins),
                self.button_val_and_continue, 
                styles={'justify' : 'around'}),
            pn.Row(self.math_env_params_ui.ui, self.rl_env_init_params_ui.ui, self.rl_env_fe_params_ui.ui),
        
            margin=main_wb_margins, styles={'border-width' : '2.0px'}, min_height=1000, min_width=1000
        
        )
        
        self.update_cart_pos_interval(event='Initialization Call')
        
    def get_param_values(self):
        return {
            'math_env_params'  : self.math_env_params_ui.get_param_values(),   
            'rl_init_params'   : self.rl_env_init_params_ui.get_param_values(),   
            'rl_env_fe_params' : self.rl_env_fe_params_ui.get_param_values(),
        }
            
    def update_on_n_change(self, event):
        n  = self.math_env_params_ui.param_to_component['n'].value
        
        self.math_env_params_ui.param_to_component['r'].value    = self.math_env_params_ui.param_to_component['r'].value[-1].repeat(n) 
        self.math_env_params_ui.param_to_component['l'].value    = self.math_env_params_ui.param_to_component['l'].value[-1].repeat(n) 
        self.math_env_params_ui.param_to_component['m'].value    = self.math_env_params_ui.param_to_component['m'].value[-1].repeat(n) 
        self.math_env_params_ui.param_to_component['\\mu'].value = self.math_env_params_ui.param_to_component['\\mu'].value[-1].repeat(n) 
        self.math_env_params_ui.param_to_component['I'].value    = self.math_env_params_ui.param_to_component['I'].value[-1].repeat(n) 
               
        self.rl_env_init_params_ui.param_to_component['equilibrium_mode'].value    = self.rl_env_init_params_ui.param_to_component['equilibrium_mode'].value[-1]*n  # repeat the last character
        self.rl_env_init_params_ui.param_to_component['initial_angles'].value      = self.rl_env_init_params_ui.param_to_component['initial_angles'].value[-1].repeat(n) 
        self.rl_env_init_params_ui.param_to_component['initial_velocities'].value  = self.rl_env_init_params_ui.param_to_component['initial_velocities'].value[-1].repeat(n) 
        
        fe_params_provider = Feature_Engineering_Param_Provider()
        #tsfe_values = self.rl_env_fe_params_ui.param_to_component['tsfe_features'].values
        self.rl_env_fe_params_ui.param_to_component['tsfe_features'].options = fe_params_provider.get_all_tsfe_features(n=n)
        #self.rl_env_fe_params_ui.param_to_component['tsfe_features'].options = fe_params_provider.get_all_tsfe_features(n=n)
        
    def update_cart_pos_interval(self, event):
        scaling_factor = 0.95
        track_width = self.math_env_params_ui.param_to_component['w'].value
        cart_width  = self.math_env_params_ui.param_to_component['w_c'].value
        bump_x = (track_width-cart_width)/2
        val_ = np.round(scaling_factor*bump_x,3)
        self.rl_env_init_params_ui.param_to_component['initial_cart_pos_interval'].value  = np.array([-val_, val_])
        
        
    def update_exp_file_info(self, exp_save_dir): 
        self.rl_env_init_params_ui.exp_save_dir = exp_save_dir # for saving the env inside the correct experiment dir
    

class Math_Env_Params_UI():
    def __init__(self):
        latex_min_width = 20
        
        self._init_mappings()
        
        self.ui = pn.WidgetBox(
            pn.pane.Markdown('## Math Environment Params', margin=wb_h2_header_margins),
            pn.layout.Divider(margin=divider_margins),
            pn.Row(self.param_to_component['env_type']),
            pn.pane.Markdown('### Constants', margin=wb_h2_header_margins),
            pn.Row(pn.pane.LaTeX('$n$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['n']),
            pn.Row(pn.pane.LaTeX('$g$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['g']),
            pn.Row(pn.pane.LaTeX('$w$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['w']),
            pn.Row(pn.pane.LaTeX('$w_c$',    min_width=latex_min_width, align='center', margin=latex_margins),    self.param_to_component['w_c']),
            pn.Row(pn.pane.LaTeX('$l$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['l']),  
            pn.Row(pn.pane.LaTeX('$r$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['r']), 
            pn.Row(pn.pane.LaTeX('$m$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['m']), 
            pn.Row(pn.pane.LaTeX('$m_c$',    min_width=latex_min_width, align='center', margin=latex_margins),    self.param_to_component['m_c']),
            pn.Row(pn.pane.LaTeX('$\\mu_c$', min_width=latex_min_width, align='center', margin=latex_margins), self.param_to_component['\\mu_c']), 
            pn.Row(pn.pane.LaTeX('$\\mu$',   min_width=latex_min_width, align='center', margin=latex_margins),   self.param_to_component['\\mu']), 
            pn.Row(pn.pane.LaTeX('$I$',      min_width=latex_min_width, align='center', margin=latex_margins),      self.param_to_component['I']),
            margin=sub_wb_margins, styles={'border-width' : '2.0px'}
        )
        
    def _init_mappings(self):
        n = 1
        self.desc_mapping = {
            'env_type'     :  "Set the environment type: \n" +\
                              "- inverted pendulum: l=r \n" +\
                              "- compound pendulum: l>r \n" +\
                              "- The compound pendulum is the only option for simplicity \n" +\
                              "- Just set l=r below to still get the inverted pendulum behaviour",
        }
        self.param_to_component = {
            'env_type'  : pn.widgets.Select(name='env type:', options=['inverted pendulum', 'compound pendulum'], value='compound pendulum', 
                                                   disabled_options=['inverted pendulum'], description=self.desc_mapping['env_type']),
            'n'         : pn.widgets.IntSlider( name='number of rods',                   value=n,end=5,start=1,step=1, bar_color=slider_bar_color),
            'g'         : pn.widgets.FloatInput(name='g (in m/s²)',                      value=9.81),
            'w'         : pn.widgets.FloatInput(name='track width (in m)',               value=1.5),
            'w_c'       : pn.widgets.FloatInput(name='cart width (in m)',                value=0.2),
            'l'         : pn.widgets.ArrayInput(name='rod lengths (in m)',               value=np.array([0.3 for i in range(n)]), ),  
            'r'         : pn.widgets.ArrayInput(name='rod center of masses (in m)',      value=np.array([0.2 for i in range(n)]), ), 
            'm'         : pn.widgets.ArrayInput(name='rod masses (in kg)',               value=np.array([0.8 for i in range(n)]), ), 
            'm_c'       : pn.widgets.FloatInput(name='cart mass (in kg, no effect)',     value=1.0),
            '\\mu_c'    : pn.widgets.FloatInput(name='cart friction (no effect)',        value=0.75, ), 
            '\\mu'      : pn.widgets.ArrayInput(name='rod frictions',                    value=np.array([0.015 for i in range(n)]), ), 
            'I'         : pn.widgets.ArrayInput(name='rod inertias (in Nms²)',           value=np.array([0.011 for i in range(n)]), ),
        }
        
    def get_param_values(self):
        n        = self.param_to_component['n'].value
        env_type = self.param_to_component['env_type'].value
        return {'env_type'  : self.param_to_component['env_type'].value,
                'constants' : {
                    'n'     : self.param_to_component['n'].value,     
                    'g'     : self.param_to_component['g'].value,   
                    'w'     : self.param_to_component['w'].value,   
                    'w_c'   : self.param_to_component['w_c'].value,    
                    'l'     : self.param_to_component['l'].value,   
                    'r'     : self.param_to_component['r'].value,   
                    'm'     : self.param_to_component['m'].value,   
                    'm_c'   : self.param_to_component['m_c'].value, 
                    '\\mu_c': self.param_to_component['\\mu_c'].value,   
                    '\\mu'  : self.param_to_component['\\mu'].value,   
                    'I'     : self.param_to_component['I'].value,               
                    },
                }

class RL_Env_Init_Params_UI():
    def __init__(self):
        self.n_ui = 1
        self.exp_save_dir = os.getcwd() # default when no experiment dir was defined
        self._init_mappings()
        
        self.episode_initializer_param_ui = \
            pn.Column(
                pn.Row(self.param_to_component['T']),
                pn.Row(self.param_to_component['initial_cart_pos']),
                pn.Row(self.param_to_component['initial_cart_velocity']),
                pn.Row(self.param_to_component['initial_angles']),
                pn.Row(self.param_to_component['initial_velocities']),
            )
            
        self.rl_episode_initializer_param_ui = \
            pn.Column(
                pn.Row(self.param_to_component['T_interval']),
                pn.Row(self.param_to_component['initial_cart_pos_interval']),
                pn.Row(self.param_to_component['initial_cart_velocity_interval']),
                pn.Row(self.param_to_component['initial_angles_interval']),
                pn.Row(self.param_to_component['initial_velocities_interval']),
            )
            
        self.episode_initializer_tabs = pn.Tabs(
                    ('Episode Initializer',        self.episode_initializer_param_ui), 
                    ('Random Episode Initializer', self.rl_episode_initializer_param_ui),
                    margin=sub_tab_margins
                    )

        self.ui = pn.WidgetBox(
            pn.pane.Markdown('## RL Env. Initialization Params', margin=wb_h2_header_margins),
            pn.layout.Divider(margin=divider_margins),
            pn.Row(self.param_to_component['dt']),
            pn.Row(self.param_to_component['max_cart_acceleration']),
            pn.Row(self.param_to_component['equilibrium_mode']), 
            
        
            
            pn.WidgetBox(
                pn.pane.Markdown('## Episode Initializer'),
                pn.Row(self.param_to_component['episode_initializer_type']), 
                self.episode_initializer_tabs,
                margin=config_sub_wb_margins, styles={'border-width' : '1.5px'}
                ),
            margin=sub_wb_margins, styles={'border-width' : '2.0px'}
        )
        self.param_to_component['episode_initializer_type'].param.watch(self.switch_tabs_on_selection, 'value')
        
    def _init_mappings(self):
        n = 1
        self.desc_mapping = {
            'episode_initializer_type' : "- Using a Random_Episode_Initializer is always recommended, as it provides much higher quality training data. \n" +\
                                         "- A normal distribution is used for sampling with a bell shape centered at the interval mean.  \n" +\
                                         "- All sampled values will lie inside the interval without outliers by using np.clip().  \n" +\
                                         "- To specify everything deterministically, chose the regular Episode_Initializer. ",
            'desc_dt'                  : "- The environment step time.",
            'desc_equilibrium'         : "- Specify an n-digit string of '0' and '1' for each rod. (1: stading, 0:hanging), examples: '11111', '0010'. \n" + \
                                         "- This equilibrium mode impacts the reward calculation; i.e. the agent tries to fully balance the rod for '11111', and else gets punished via reward. \n" +\
                                         "- The rightmost digits are for the lowest rods",
            'desc_T'                   : "- The base episode sampling interval for the Random_Episode_Initializer.",
            'desc_angles'              : "- Specify the net rotation angles (from -inf to +inf) \n " +\
                                         "- An angle is 0|+2pi|-2pi|... when pointing up \n" + \
                                         "- An angle is pi|+3pi|-pi|... when hanging down \n" + \
                                         "- An angle is -pi/2|... when pointing right \n" + \
                                         "- An angle is pi/2|... when pointing left"            
        }
        
        self.param_to_component = {
            'dt'                             : pn.widgets.FloatInput(name='dt (in s)',                                 value=0.001, start=0.00001, end=0.01, description=self.desc_mapping['desc_dt']),
            'max_cart_acceleration'          : pn.widgets.FloatInput(name='max_cart_acceleration (in m/s²)',           value=15.0, start=0),
            'equilibrium_mode'               : pn.widgets.TextInput( name='equilibrium_mode',                          value= f"{'1'*self.n_ui}", description=self.desc_mapping['desc_equilibrium']), 
            'episode_initializer_type'       : pn.widgets.Select(    name='episode_initializer_type',                  value='Episode_Initializer', options=['Episode_Initializer', 'Random_Episode_Initializer'], description=self.desc_mapping['episode_initializer_type']),
            
            'T'                              : pn.widgets.FloatInput(name='T (in s)',                                  value=10, start=1, description=self.desc_mapping['desc_T']),
            'initial_cart_pos'               : pn.widgets.FloatInput(name='initial_cart_pos (in m)',                   value=0.0),
            'initial_cart_velocity'          : pn.widgets.FloatInput(name='initial_cart_velocity (in m/s)',            value=0.0),
            'initial_angles'                 : pn.widgets.ArrayInput(name='initial_angles (in rad)',                   value=np.array([np.round(np.pi,4) for i in range(self.n_ui)]), description=self.desc_mapping['desc_angles']),
            'initial_velocities'             : pn.widgets.ArrayInput(name='initial_velocities (in rad/s)',             value=np.array([0 for i in range(self.n_ui)]),),
            
            'T_interval'                     : pn.widgets.ArrayInput(name='T interval (in s)',                         value=np.array([9.0, 11.0]), description=self.desc_mapping['desc_T']),
            'initial_cart_pos_interval'      : pn.widgets.ArrayInput(name='initial_cart_pos interval (in m)',          value=np.array([0, 0])), # value is set by callback
            'initial_cart_velocity_interval' : pn.widgets.ArrayInput(name='initial_cart_velocity interval (in m/s)',   value=np.array([-2.0, 2.0])),
            'initial_angles_interval'        : pn.widgets.ArrayInput(name='initial_angles interval (in rad)',          value=np.array([-5*np.round(np.pi,4), 5*np.round(np.pi,4)]), description=self.desc_mapping['desc_angles']),
            'initial_velocities_interval'    : pn.widgets.ArrayInput(name='initial_velocities interval (in rad/s)',    value=np.array([-35, 35])),
        }
        

    def get_param_values(self):
        n = len(self.param_to_component['initial_angles'].value) # no callback needed: whenever n changes, the list
        env_type = 'compound pendulum' # no callback needed: is the only valid option nonetheless
        env_initializer_type = self.param_to_component['episode_initializer_type'].value
        return {
            'save_after_init'                : True, # can/should always left True: will save the environment object after its long initialization (file caching)
            'save_filename'                  : f"RL_env_n{n}_{env_type.replace(' ', '_')}_obj",
            'save_dirname'                   : self.exp_save_dir,
            
            'dt'                             : self.param_to_component['dt'].value,
            'max_cart_acceleration'          : self.param_to_component['max_cart_acceleration'].value,
            'equilibrium_mode'               : self.param_to_component['equilibrium_mode'].value,
            
            'episode_initializer_type'       : env_initializer_type,
            
            'T'                              : self.param_to_component['T'].value                                if env_initializer_type == 'Episode_Initializer' else None,
            'initial_cart_pos'               : self.param_to_component['initial_cart_pos'].value                 if env_initializer_type == 'Episode_Initializer' else None,
            'initial_cart_velocity'          : self.param_to_component['initial_cart_velocity'].value            if env_initializer_type == 'Episode_Initializer' else None,
            'initial_angles'                 : self.param_to_component['initial_angles'].value                   if env_initializer_type == 'Episode_Initializer' else None,
            'initial_velocities'             : self.param_to_component['initial_velocities'].value               if env_initializer_type == 'Episode_Initializer' else None,
            
            'T_interval'                     : self.param_to_component['T_interval'].value                       if env_initializer_type == 'Random_Episode_Initializer' else None,
            'initial_cart_pos_interval'      : self.param_to_component['initial_cart_pos_interval'].value        if env_initializer_type == 'Random_Episode_Initializer' else None,
            'initial_cart_velocity_interval' : self.param_to_component['initial_cart_velocity_interval'].value   if env_initializer_type == 'Random_Episode_Initializer' else None,
            'initial_angles_interval'        : self.param_to_component['initial_angles_interval'].value          if env_initializer_type == 'Random_Episode_Initializer' else None,
            'initial_velocities_interval'    : self.param_to_component['initial_velocities_interval'].value      if env_initializer_type == 'Random_Episode_Initializer' else None,
        }
        
    def switch_tabs_on_selection(self, event):
        if self.param_to_component['episode_initializer_type'].value == 'Episode_Initializer':
            self.episode_initializer_tabs.active = 0
        elif self.param_to_component['episode_initializer_type'].value == 'Random_Episode_Initializer':
            self.episode_initializer_tabs.active = 1
                
class RL_Env_FE_Params_UI():
    def __init__(self):

        self._init_mappings()
        self.ui = pn.WidgetBox(
            pn.pane.Markdown('## Feature Engineering Params', margin=wb_h2_header_margins),
            pn.layout.Divider(margin=divider_margins),
            pn.Row(*[
                pn.Column(
                  *[pn.Row(self.param_to_component['do_feature_engineering']),
                    pn.pane.Markdown('### Classical Feature Engineering'),
                    pn.Row(self.param_to_component['do_classic_engineering']),
                    pn.Row(self.param_to_component['feature_groups']),
                    pn.pane.Markdown('### Derivative Feature Engineering'),
                    pn.Row(self.param_to_component['do_derivative_engineering']),
                    pn.Row(self.param_to_component['dfe_feature_groups']),
                    pn.Row(self.param_to_component['dfe_included_base_features']),
                    pn.Row(self.param_to_component['max_derivative_order'])]
                ),
                pn.Column(
                  *[pn.pane.Markdown('### Time Series Feature Engineering'),
                    pn.Row(self.param_to_component['do_ts_feature_engineering']),
                    pn.Row(self.param_to_component['tsfe_lookback_mode']),
                    pn.Row(self.param_to_component['tsfe_features']),
                    pn.Row(self.param_to_component['do_difference_tsfe'],
                           pn.widgets.TooltipIcon(value=self.desc_mapping['do_diff_tsfe'], margin=(0,0,0,0))),
                    
                    pn.Row(self.param_to_component['do_overlapping_win_tsfe'],
                           pn.widgets.TooltipIcon(value=self.desc_mapping['do_overl_tsfe'], margin=(0,0,0,0))),
                    pn.Row(self.param_to_component['overlapping_fwin_aggfuncs']),
                    pn.Row(self.param_to_component['overlapping_now_fwin_aggfuncs']),
                    pn.Row(self.param_to_component['overlapping_ffwin_fwin_aggfuncs']),
                    
                    pn.Row(self.param_to_component['do_non_overlapping_win_tsfe'],
                           pn.widgets.TooltipIcon(value=self.desc_mapping['do_non_overl_tsfe'], margin=(0,0,0,0))),
                    pn.Row(self.param_to_component['non_overlapping_fwin_aggfuncs']),
                    pn.Row(self.param_to_component['non_overlapping_now_fwin_aggfuncs']),
                    pn.Row(self.param_to_component['non_overlapping_ffwin_fwin_aggfuncs']),]
                ),
            ]), margin=sub_wb_margins, styles={'border-width' : '2.0px'}
        )
            
    def _init_mappings(self):
        n = 1
        fe_params_provider = Feature_Engineering_Param_Provider()
        all_feat_groups = fe_params_provider.get_all_feature_groups()
        all_tsfe_feats = fe_params_provider.get_all_tsfe_features(n=n)
        
        self.desc_mapping = {
            # CFE
            'do_cfe'                    : "- Disable the complete process of classic feature engineering (CFE). \n " + \
                                          "- This also disables DFE and TSFE which are dependendent on CFE.",
            'feat_groups'               : "- Select feature groups for classical feature engineering.",
            # DFE
            'dfe_feat_groups'           : "- Select feature groups to be differentiated by t inside DFE. \n" + \
                                          "- These feature groups also have to be in feature_groups! \n " + \
                                          "- Base features are not differentiated (they contain time derivatives already) \n",
            'dfe_base_feats'            : "- Include potentially interesting base feature into DFE \n" + \
                                          "- These include the continuous reward and action here. \n " ,
            'max_derivative_order'      : "- Specify the maximum order of differentiation to differentiate all DFE feature groups. \n" +\
                                          "- Warning: Whenever noise is involved, choose at max the derivative order of 1.",
            # TSFE
            'tsfe_lookback_wins'        : "- Select between ['long', 'medium', 'short'] to shorten the maximum lookback window of 2s and also to reduce the amount of features \n" +\
                                          "- 'long' : 2s; 'medium' 0.5s; 'short' 0.125s \n" +\
                                          "- 'long' : 100% of features; 'medium' ~50-60% of features; 'short' ~ 25-35% of features",
            'tsfe_features'             : "- Chose features that will be part of TSFE \n" + \
                                          "- You may only chose around 2-5 features for small models (there will be a lot of output features) \n" + \
                                          "- Note: these features explicitly represent the column names of output features. They look like '˙xc\dot{x_c}'",
            'do_diff_tsfe'              : "- Toggle the subprocess of TSFE which involves taking differences between two time stamps. \n" +\
                                          "- Differences are applied to all selected features on a predefined offset window.",
            'do_overl_tsfe'             : "- Toggle the subprocess of TSFE which involves applying aggregation functions on overlapping windows. \n" + \
                                          "- All aggregation windows all stretch back n steps starting from the most recent time step. \n" +\
                                          "- Thus all capture the most recent data, but with a different amounts of past data.",
            'do_non_overl_tsfe'         : "- Toggle the subprocess of TSFE which involves applying aggregation functions on non-overlapping windows. \n" + \
                                          "- All aggregation windows lie consecutively behind each other without gaps or overlapping. \n" +\
                                          "- Each window starts behind the previous window capturing separate parts of the time series.",
            'overl_fwin_aggfuncs'       : "- Chose a set of aggregation functions to capture specific information within each aggregated lookback window. \n" +\
                                          "- This function is applied like 'f_win' (Simple lookback aggregation window).",
            'overl_now_fwin_aggfuncs'   : "- Chose a set of aggregation functions to capture specific information within each aggregated lookback window. \n" +\
                                          "- This function is applied like 'now-f_win' (subtract lookback aggregation windows from current time step).",
            'overl_ffwin_fwin_aggfuncs' : "- Chose a combination of aggregation functions to capture specific information within each aggregated lookback window. \n" +\
                                          "- This function is applied like 'ff_win - f_win' (difference between first and later lookback aggragation windows).",
        }
        
        self.param_to_component = {
            'do_feature_engineering'              : pn.widgets.Checkbox(   name='do_feature_engineering',              value=False),
            # CFE
            'do_classic_engineering'              : pn.widgets.Checkbox(   name='do_classic_engineering',              value=False),
            'feature_groups'                      : pn.widgets.MultiChoice(name='feature_groups',                      value=[],     description=self.desc_mapping['feat_groups'],           options=all_feat_groups,             ), 
            # DFE
            'do_derivative_engineering'           : pn.widgets.Checkbox(   name='do_derivative_engineering',           value=False),
            'dfe_feature_groups'                  : pn.widgets.MultiChoice(name='dfe_feature_groups',                  value=[],     description=self.desc_mapping['dfe_feat_groups'],       options=all_feat_groups,             ),
            'dfe_included_base_features'          : pn.widgets.MultiChoice(name='dfe_included_base_features',          value=[],     description=self.desc_mapping['dfe_base_feats'],        options=['$$r$$', r'$$\ddot{x_c}$$'],),
            'max_derivative_order'                : pn.widgets.Select(     name='max_derivative_order',                value=1,      description=self.desc_mapping['max_derivative_order'],  options=[1,2],                       ), 
            # TSFE
            'do_ts_feature_engineering'           : pn.widgets.Checkbox(   name='do_ts_feature_engineering',           value=False),
            'tsfe_lookback_mode'                  : pn.widgets.Select(     name='tsfe_lookback_mode',                  value='long', description=self.desc_mapping['tsfe_lookback_wins'],    options=['long', 'medium', 'short'], ),
            'tsfe_features'                       : pn.widgets.MultiChoice(name='tsfe_features',                       value=[],     description=self.desc_mapping['tsfe_features'],         options=all_tsfe_feats,              ),
            'do_difference_tsfe'                  : pn.widgets.Checkbox(   name='do_difference_tsfe',                  value=False),
            
            'do_overlapping_win_tsfe'             : pn.widgets.Checkbox(   name='do_overlapping_win_tsfe',             value=False), 
            'overlapping_fwin_aggfuncs'           : pn.widgets.MultiChoice(name='overlapping_fwin_aggfuncs',           value=[],     description=self.desc_mapping['overl_fwin_aggfuncs'],      options=['Mean', 'Min', 'Max', 'Std'],     ),
            'overlapping_now_fwin_aggfuncs'       : pn.widgets.MultiChoice(name='overlapping_now_fwin_aggfuncs',       value=[],     description=self.desc_mapping['overl_now_fwin_aggfuncs'],  options=['Now-Mean', 'Now-Min', 'Now-Max'],),
            'overlapping_ffwin_fwin_aggfuncs'     : pn.widgets.MultiChoice(name='overlapping_ffwin_fwin_aggfuncs',     value=[],     description=self.desc_mapping['overl_ffwin_fwin_aggfuncs'],    
                                                                           options=['Mean-Mean', 'Mean-Min', 'Mean-Max', 'Min-Mean',  'Min-Min',  'Min-Max','Max-Mean',  'Max-Min',  'Max-Max','Std-Std']),
            
            'do_non_overlapping_win_tsfe'         : pn.widgets.Checkbox(   name='do_non_overlapping_win_tsfe',         value=False),
            'non_overlapping_fwin_aggfuncs'       : pn.widgets.MultiChoice(name='non_overlapping_fwin_aggfuncs',           value=[],     description=self.desc_mapping['overl_fwin_aggfuncs'],      options=['Mean', 'Min', 'Max', 'Std'],     ),
            'non_overlapping_now_fwin_aggfuncs'   : pn.widgets.MultiChoice(name='non_overlapping_now_fwin_aggfuncs',       value=[],     description=self.desc_mapping['overl_now_fwin_aggfuncs'],  options=['Now-Mean', 'Now-Min', 'Now-Max'],),
            'non_overlapping_ffwin_fwin_aggfuncs' : pn.widgets.MultiChoice(name='non_overlapping_ffwin_fwin_aggfuncs',     value=[],     description=self.desc_mapping['overl_ffwin_fwin_aggfuncs'],  
                                                                           options=['Mean-Mean', 'Mean-Min', 'Mean-Max', 'Min-Mean',  'Min-Min',  'Min-Max','Max-Mean',  'Max-Min',  'Max-Max','Std-Std']),
        }

    def get_param_values(self):
        return {
            'do_feature_engineering'                   : self.param_to_component['do_feature_engineering'].value,
            'CFE' : {
                'do_classic_engineering'               : self.param_to_component['do_classic_engineering'].value,
                'feature_groups'                       : self.param_to_component['feature_groups'].value,
            },
            'DFE' : {
                'do_derivative_engineering'            : self.param_to_component['do_derivative_engineering'].value,
                'dfe_feature_groups'                   : self.param_to_component['dfe_feature_groups'].value,
                'dfe_included_base_features'           : self.param_to_component['dfe_included_base_features'].value,
                'max_derivative_order'                 : self.param_to_component['max_derivative_order'].value,            
            },
            'TSFE' : {
                'do_ts_feature_engineering'            : self.param_to_component['do_ts_feature_engineering'].value,
                'tsfe_lookback_mode'                   : self.param_to_component['tsfe_lookback_mode'].value,
                'tsfe_features'                        : self.param_to_component['tsfe_features'].value,
                'do_difference_tsfe'                   : self.param_to_component['do_difference_tsfe'].value,
                
                'do_overlapping_win_tsfe'              : self.param_to_component['do_overlapping_win_tsfe'].value,
                'overlapping_fwin_aggfuncs'            : self.param_to_component['overlapping_fwin_aggfuncs'].value,
                'overlapping_now_fwin_aggfuncs'        : self.param_to_component['overlapping_now_fwin_aggfuncs'].value,
                'overlapping_ffwin_fwin_aggfuncs'      : self.param_to_component['overlapping_ffwin_fwin_aggfuncs'].value,
                
                'do_non_overlapping_win_tsfe'          : self.param_to_component['do_non_overlapping_win_tsfe'].value,
                'non_overlapping_fwin_aggfuncs'        : self.param_to_component['non_overlapping_fwin_aggfuncs'].value,
                'non_overlapping_now_fwin_aggfuncs'    : self.param_to_component['non_overlapping_now_fwin_aggfuncs'].value,
                'non_overlapping_ffwin_fwin_aggfuncs'  : self.param_to_component['non_overlapping_ffwin_fwin_aggfuncs'].value,          
            },
        }

    
