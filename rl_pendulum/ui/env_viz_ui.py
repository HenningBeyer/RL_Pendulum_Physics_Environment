import numpy as np
import panel as pn
import pandas as pd
import time
import plotly_express as px
import os

from rl_pendulum.ui.ui_plotting import get_empty_ts_plot_figure, update_ts_plot_figure, update_ts_plot_index_range,\
                                       get_empty_pc_figure, update_pc_plot,\
                                       get_empty_kde_plot_figure, update_kde_plot
from rl_pendulum.ui.rl_config_ui import RL_Env_Init_Params_UI
from rl_pendulum.utils.file_utils import find_new_numeric_file_name
from rl_pendulum.ui.styling import wb_h2_header_margins, divider_margins, latex_margins, \
                                   slider_bar_color, wb_h1_header_margins,\
                                   wb_h1_tooltip_margins, wb_h2_tooltip_margins, main_wb_margins, sub_wb_margins,\
                                   sub_tab_margins, panel_side_wb_margins, df_width, side_wb_width, side_wb_component_width,\
                                    button_h1_margins, button_h2_margins
 
def get_dummy_df():
    cols = [f'c{i}' for i in range(100)]
    return pd.DataFrame(np.random.rand(100,100), columns=cols) # get a df with random values as placeholder for plotting, etc.                                  

class Env_Data_Analysis_UI:
    def __init__(self):
        self._init_mappings()
        self.button_initialize_env     = pn.widgets.Button(name="Initialize Environment", margin=button_h1_margins)
        self.button_calculate_sim_data = pn.widgets.Button(name="Calculate Simulation Data",margin=button_h1_margins)
        self.button_continue           = pn.widgets.Button(name="Continue", margin=button_h1_margins)
        
        self.rl_sim_env = None # this is initialized by button_initialize_env; callback defined in main_ui.py
        self.env_data = get_dummy_df()


        self.sim_parameter_wb       = Env_Simulation_Init_Params_UI()
        self.sim_data_preview_panel = Data_Preview_Panel()
        self.sim_data_viz_panel     = Data_Analysis_Panel()
        self.ui = pn.WidgetBox(pn.Row(pn.pane.Markdown('# Environment Data Analysis', margin=wb_h1_header_margins), 
                                   pn.widgets.TooltipIcon(value=self.desc_mapping['general_info'], margin=wb_h1_tooltip_margins),
                                   #pn.widgets.TooltipIcon(value=self.desc_mapping['note_info'], margin=wb_h1_tooltip_margins),
                                   self.button_initialize_env, 
                                   self.button_calculate_sim_data, 
                                   self.button_continue,
                                   ),
                               pn.Row(
                                   pn.Column(self.sim_parameter_wb.ui),
                                   pn.Column(self.sim_data_preview_panel.ui, self.sim_data_viz_panel.ui)
                                ), margin=main_wb_margins, styles={'border-width' : '2.0px'}, min_width=1540, min_height=1500 # sizing both width and height is not hardly possible: have to specify exact min_width and max_width to allow sizing on zooming-in AND zooming-out
        )

    def _init_mappings(self):
        self.desc_mapping = {
            'general_info'      : "- This section is an optional but recommended checkpoint to review all your data \n" +\
                                  "- Also use this to gain some more in-depth insights to your data. \n" +\
                                  "- The parameters below will have no effect later on.",
        }
        
    def update_data(self, df, info):
        """ This function is called after pressing 'Calculate Simulation Data'.
            Out of convenience, the entire UI is re-initialied, which avoids multiple callbacks
        """
        self.env_data = df
        self.sim_data_preview_panel.update_data(self.env_data) # this changes the underlying ui attribute, changing the self.ui from Env_Data_Analysis_UI too
        if info != None:
            self.sim_data_preview_panel.update_step_time_info(info)
        self.sim_data_viz_panel.update_data(self.env_data)
        
    def update_exp_file_info(self, exp_save_dir): 
        self.sim_data_preview_panel.exp_save_dir = exp_save_dir
        self.sim_data_viz_panel.exp_save_dir = exp_save_dir
        
        
class Env_Simulation_Init_Params_UI:
    """ This class reuses RL_Env_Init_Params_UI and just extends it by a few components as the UI remains very similar. """
    def __init__(self):
        self._init_mappings()
        
        # RL_Env_Init_Params_UI() cant be reused, as the mandatory references between self.ui and self.param_to_component will break if altering them separately
        # This causes callbacks not to affect one of the attributes, which is very hard to find out for debugging.
        self.ui = pn.WidgetBox( 
            pn.pane.Markdown('## Env. Simulation Parameters', margin=wb_h2_header_margins),
            pn.layout.Divider(margin=divider_margins),
            pn.Row(self.param_to_component['save_env'],
                   pn.widgets.TooltipIcon(value=self.desc_mapping['save_env'], margin=(0,0,0,0))),
            pn.Row(pn.widgets.StaticText(value='Load Environment', margin=(0,0,0,10)),
                   pn.widgets.TooltipIcon(value=self.desc_mapping['load_env'], margin=(0,0,0,0))),
            pn.Row(self.param_to_component['load_env']),
            pn.Row(pn.widgets.StaticText(value='Load Dataset', margin=(0,0,0,10)),
                   pn.widgets.TooltipIcon(value=self.desc_mapping['load_dataset'], margin=(0,0,0,0))),
            pn.Row(self.param_to_component['load_dataset']),
            pn.Row(self.param_to_component['control_func']),
            margin=sub_wb_margins, styles={'border-width' : '2.0px'}
        )

    def _init_mappings(self):
        self.n_ui = 1
        
        self.desc_mapping = {
            'save_env'           : "- The initialized environment will always be saved to be loaded faster later.",
            'load_env'           : "- Load any RL_Cart_Pendulum_Environment Python object from a .pkl file to avoid long initialization times. \n" +\
                                   "- The math constants may differ from your configuration, but all other parameters from your configuration will be applied. \n" +\
                                   "- The n parameter from your object and your configuration have to match. \n" +\
                                   "- SECURITY WARNING: .pkl files can contain arbitrary Python scripts that are run after loading. Only use your own .pkl files. ",
            'load_dataset'       : "- Load a .csv dataset to load saved simulation data. \n" +\
                                   "- Any arbitrary dataset should work too.",
            'desc_control_func'  : "- Select a mathmatical control function to get data closer to the behaviour of an RL agent. "
        }
        
        self.str_to_control_func = {
            '0' : lambda t: 0, 
            'sin(4t)/2' : lambda t: 0.5*np.sin(t*4),
            'sin(t)/4 + cos(2t)/4 + cos(4t)/4' : lambda t: 0.25*np.sin(t) + 0.25*np.cos(2*t) + 0.25*np.cos(4*t)
        }
                
        self.param_to_component = {
            'save_env'      : pn.widgets.Checkbox(name='Save Initialized Environment', value=True, disabled=True), # always do an auto save; but show this to the user
            'load_env'      : pn.widgets.FileInput(accept='.pkl', multiple=False, margin=(0,0,10,10)),
            'load_dataset'  : pn.widgets.FileInput(accept='.csv', multiple=False, margin=(0,0,10,10)),
            'control_func'  : pn.widgets.Select(name='Simulation Control Function', value='0', description=self.desc_mapping['desc_control_func'], options=list(self.str_to_control_func.keys())),
        }
        
    def get_rl_init_param_values(self):
        pass
    
        n = len(self.param_to_component['initial_angles'].value) # no callback needed: whenever n changes, the list
        env_type = 'compound pendulum' # no callback needed: is the only valid option nonetheless
        return {
            'control_func' :   self.param_to_component['control_func'].value
        }
        
class Data_Preview_Panel:
    def __init__(self):
        self.env_data = get_dummy_df()
        self.exp_save_dir = os.getcwd()
        self.file_prefix  = 'episode_data_sample_'
        self._init_mappings()

        self.button_download_dataset = pn.widgets.Button(name="Save Dataset", margin=button_h1_margins)
        self.df_preview = pn.pane.DataFrame(self.env_data, max_rows=11, max_width=df_width, margin=(20,10,10,20))
        self.df_summary = pn.pane.DataFrame(self.get_df_summary(self.env_data), max_rows=11, max_width=df_width, margin=(20,10,10,20))
        self.stat_text1 = \
                f"""
                 ### General Stats:
                 * Input features: {len(self.env_data.columns)}
                 * Time steps: {len(self.env_data)}
                 * Total NaN values: {np.sum(np.sum(self.env_data.isna(), axis=0))}
                 * RAM consumption: {np.round(self.env_data.memory_usage(index=True, deep=True).sum()/10**6,3)} MB
                 """
        self.stat_text2 = \
                f"""
                 ### Step Times:
                 * Env. simulation step: NaN
                 * Env. feature engineering: NaN
                 * Agent step time: NaN
                 * Total: NaN
                 """
        self.stat_md1  = pn.pane.Markdown(self.stat_text1)
        self.stat_md2  = pn.pane.Markdown(self.stat_text2)
        self.wb_slider = pn.widgets.IntRangeSlider(name='Index Range', start=self.env_data.index[0], end=self.env_data.index[-1], step=1, bar_color=slider_bar_color, width=side_wb_component_width)
        
        ### Callbacks
        self.wb_slider.param.watch(self.range_slider_callback_tables, 'value')
        self.button_download_dataset.on_click(self.save_dataset)
        ###
        
        self.side_wb = pn.layout.WidgetBox(
            pn.pane.Markdown('### Value Range Inspection:'),
            self.wb_slider,
            self.stat_md1,
            self.stat_md2,
          margin=panel_side_wb_margins, width=side_wb_width, styles={'border-width' : '2.0px'}, align=('end', 'start')
        )
        
        ui_width = 1150
        
        # setting a lot of self attributes to let changes on self.df_preview eventually affect self.ui
        self.tabular_preview_pane = pn.Row(
            self.df_preview,
            pn.HSpacer(),
            self.side_wb,
            max_width=ui_width) # min width based on df np nan, max_ width is based on a full df with all rows
        self.data_summary_pane = pn.Row(
            self.df_summary,
            pn.HSpacer(),
            self.side_wb,
            max_width=ui_width
            ) # the two instances of side_wb are synchronized; share the same reference

        self.ui = pn.layout.WidgetBox(
            pn.Row(pn.pane.Markdown('# Data Preview', margin=wb_h1_header_margins), 
                   pn.widgets.TooltipIcon(value=self.desc_mapping['general_info'], margin=wb_h1_tooltip_margins),
                   self.button_download_dataset),
            pn.Row(pn.layout.Tabs(('Tabular Preview',self.tabular_preview_pane),
                                  ('Data Summary',self.data_summary_pane), 
                                  width=ui_width,
                                  dynamic=False, margin=sub_tab_margins, styles={'border-width' : '2.0px'})), 
            margin=sub_wb_margins, styles={'border-width' : '2.0px'}, width=ui_width+15
            )
        
    def update_data(self, df):
        """ Reset the entire Data_Preview_Panel with calculated DataFrame data.
            This function is way more convenient than multiple callbacks: everything is in one function
            
            Note that self.env_data must be set by another class before updating the components
        """
        self.env_data = df # This is set by reference
        self.df_preview.object = self.env_data
        self.df_summary.object = self.get_df_summary(self.env_data)
        self.stat_md1.object = f"""
                                ### General Stats:
                                * Input features: {len(self.env_data.columns)}
                                * Time steps: {len(self.env_data)}
                                * Total NaN values: {np.sum(np.sum(self.env_data.isna(), axis=0))}
                                * RAM consumption: {np.round(self.env_data.memory_usage(index=True, deep=True).sum()/10**6,3)} MB
                                """
                                
        self.wb_slider.start = 0
        self.wb_slider.step  = 1
        self.wb_slider.end   = len(self.env_data)-1
        self.wb_slider.value = (self.wb_slider.start, self.wb_slider.end)
                                  
    def update_step_time_info(self, info : dict):
        self.stat_md2.object = f"""                    
                                ### Step Times:
                                * Env. simulation step: {np.round(info['env'],4 )} s
                                * Env. feature engineering: {np.round( info['fe'], 4 )} s
                                * Agent step time: {np.round( info['agent'], 4 )} s
                                * Total: {np.round( info['env'] + info['fe'] + info['agent'], 4 )} s 
                                """
        

    def get_df_summary(self, df): # callable after 'Calculate Simulation Data' was pressed
        return df.describe(percentiles=[0.025,0.25,0.5,0.75,0.975])
        
    def range_slider_callback_tables(self, event):
        start, end = self.wb_slider.value[0], self.wb_slider.value[1]
        self.df_preview.object = self.env_data.iloc[start:end+1]
        self.df_summary.object = self.get_df_summary(self.env_data.iloc[start:end+1])
        
    def _init_mappings(self):
        self.desc_mapping = {
            'general_info' : """- View the dataset in tabular form to check the general dataset information."""
        }
        
    def save_dataset(self, event):
        file_name = find_new_numeric_file_name(base_dir=self.exp_save_dir, file_prefix=self.file_prefix, file_extension='.csv') # --> sample_dataset_3.csv
        self.env_data.to_csv(self.exp_save_dir + '\\' + file_name)
        
            

        
        
class Data_Analysis_Panel:
    def __init__(self):
        self.env_data = get_dummy_df() # self.env_data will be overwritten when used as subclass
        self.exp_save_dir = os.getcwd()
        self._init_mappings()
        self.button_save_opened_plot = pn.widgets.Button(name="Save Figure", margin=button_h1_margins)
        
        self.wb_slider           = pn.widgets.IntRangeSlider(name='Index Range', start=self.env_data.index[0], end=self.env_data.index[-1], step=1, bar_color=slider_bar_color, width=side_wb_component_width)
        self.button_sync_feats   = pn.widgets.Button(name='Syncronize Feature Selection', description="- Selects the current feature selection from this tab for all other tabs as well.", value=True)
        
        self.ts_plot_feats       = pn.widgets.MultiChoice(name='Plotted Features', value=list(self.env_data.columns[:4]), options=list(self.env_data.columns), width=side_wb_component_width) 
        self.pc_plot_feats       = pn.widgets.MultiChoice(name='Parallel Features', value=list(self.env_data.columns[:4]), options=list(self.env_data.columns), width=side_wb_component_width)
        self.pc_coloring_feats   = pn.widgets.Select(name='Parallel Coloring Feature', value=self.env_data.columns[0], options=self.pc_plot_feats.value, width=side_wb_component_width)
        self.kde_plot_feats      = pn.widgets.MultiChoice(name='KDE Features', value=list(self.env_data.columns[:4]), options=list(self.env_data.columns), width=side_wb_component_width)
        
        self.band_width_slider   = pn.widgets.FloatSlider( name='band_width scaling', value=1.0,end=1.25,start=0.1,step=0.01, bar_color=slider_bar_color, width=side_wb_component_width)
        self.feat_scaling_method = pn.widgets.Select(name='Feature Scaling Method', value='No Scaling', options=['No Scaling', 'Standard Scaling', 'Min-Max Scaling'], width=side_wb_component_width)
     
        ### Callbacks
        self.wb_slider.param.watch(self.range_slider_callback, 'value') # updates on any slider change
        self.wb_slider.param.watch(self.range_slider_callback_throttled, 'value_throttled') # updates on slider release
        self.feat_scaling_method.param.watch(self.update_feat_scaling, 'value') 
        self.button_sync_feats.on_click(self.update_syncronized_feature_selection)
        self.button_save_opened_plot.on_click(self.save_figure)
        
        self.ts_plot_feats.param.watch(     self.update_ts_plot_features,  'value')
    
        self.pc_plot_feats.param.watch(     self.update_pc_plot,  'value')
        self.pc_coloring_feats.param.watch( self.update_pc_plot,  'value')
        
        self.kde_plot_feats.param.watch(    self.update_kde_plot, 'value')
        self.band_width_slider.param.watch( self.update_kde_plot, 'value') 
        ###
        
        self.side_wb_ts = pn.layout.WidgetBox(
            pn.pane.Markdown('### Value Range Inspection:'),
            self.wb_slider,
            pn.pane.Markdown('### Feature Inspection:'),
            self.button_sync_feats,
            self.ts_plot_feats,
            self.feat_scaling_method,
            margin=panel_side_wb_margins, width=side_wb_width, styles={'border-width' : '2.0px'}, align=('end', 'start')
        )
        
        self.side_wb_pc = pn.layout.WidgetBox(
            pn.pane.Markdown('### Value Range Inspection:'),
            self.wb_slider, # the instances of wb_slider are now synchronized for each tab
            pn.pane.Markdown('### Feature Inspection:'),
            self.button_sync_feats,
            self.pc_plot_feats,
            self.pc_coloring_feats,
            margin=panel_side_wb_margins, width=side_wb_width, styles={'border-width' : '2.0px'}, align=('end', 'start')
        )
        
        self.side_wb_kde = pn.layout.WidgetBox(
            pn.pane.Markdown('### Value Range Inspection:'),
            self.wb_slider, # the instances of wb_slider are now synchronized for each tab
            pn.pane.Markdown('### Feature Inspection:'),
            self.button_sync_feats,
            self.kde_plot_feats,
            self.band_width_slider,
            self.feat_scaling_method,
            margin=panel_side_wb_margins, width=side_wb_width, styles={'border-width' : '2.0px'}, align=('end', 'start')
        )
        
        ui_width = 1150
        idx_range_ = self.wb_slider.value
        fig_ts_plot = get_empty_ts_plot_figure()
        fig_ts_plot = update_ts_plot_figure(fig=fig_ts_plot, 
                                            df=self.env_data, 
                                            col_names=self.env_data.columns[:4],
                                            scaling_method=self.feat_scaling_method.value) # updates fig_ts_plot internally, but returns it, in case of a reset 
        fig_pc_plot = update_pc_plot(df=self.env_data, 
                                     col_names=self.env_data.columns[:4], 
                                     color_column=self.pc_coloring_feats.value, 
                                     idx_range=self.wb_slider.value) # updates fig_pc_plot internally 
        fig_kde_plot = update_kde_plot(df=self.env_data, 
                                       col_names=self.env_data.columns[:4],
                                       idx_range=self.wb_slider.value,
                                       bw_adjust=self.band_width_slider.value,
                                       scaling_method=self.feat_scaling_method.value)
                                               
        
        self.ts_plot  = pn.pane.Plotly(fig_ts_plot,  margin=(20,10,10,20))
        self.pc_plot  = pn.pane.Plotly(fig_pc_plot,  margin=(20,10,10,20))
        self.kde_plot = pn.pane.Plotly(fig_kde_plot, margin=(20,10,10,20))
        
        self.tabular_preview_pane = pn.Row(
            self.ts_plot,
            pn.HSpacer(),
            self.side_wb_ts,
            width=ui_width
        )
        
        self.parallel_coordinates_pane = pn.Row(
            self.pc_plot,
            pn.HSpacer(),
            self.side_wb_pc,
            width=ui_width) # the instances of side_wb are now synchronized
        
        self.kde_plot_pane = pn.Row(
            self.kde_plot,
            pn.HSpacer(),
            self.side_wb_kde,
            width=ui_width) # the instances of side_wb are now synchronized
        
        self.tabs = pn.layout.Tabs(('Time Series Plots',   self.tabular_preview_pane),
                                   ('Parallel Coordinates',self.parallel_coordinates_pane), 
                                   ('KDE Plots',self.kde_plot_pane), 
                               min_width=side_wb_width, dynamic=False, margin=sub_tab_margins, styles={'border-width' : '2.0px'})

        self.ui = pn.layout.WidgetBox(
            pn.Row(pn.pane.Markdown('# Data Analysis', margin=wb_h1_header_margins),
                   pn.widgets.TooltipIcon(value=self.desc_mapping['general_info'], margin=wb_h1_tooltip_margins),
                   #pn.widgets.TooltipIcon(value=self.desc_mapping['note_info'], margin=wb_h1_tooltip_margins),
                   self.button_save_opened_plot
                   ),
            pn.Row(self.tabs), 
            margin=sub_wb_margins, styles={'border-width' : '2.0px'}, width=ui_width+15
            )

    def _init_mappings(self):
        self.desc_mapping = {
            'general_info' : "- Visualize any time series feature via plotting. \n" +\
                             "- Observe dataset-wise and time-dependent feature correlations via parallel coordinates. \n" +\
                             "- Get a general idea how the underlying feature distributions look like by viewing KDE plots.",
            'note_info'    : "- You can also look at math.ipynb to see rendered animations for the environment. \n" +\
                             "- You can also animate your own downloaded data with the same notebook, or later your agent's final results. \n" #+\
                             #"- All animations take minutes to render, which is too slow, and they are not fully supported by Panel."
        }
        
    def update_data(self, df : pd.DataFrame):
        """ Reset the entire Data_Preview_Panel with calculated DataFrame data.
            This function is way more convenient than multiple callbacks: everything is in one function
        """
        self.env_data = df # this is set by reference when passing a df
        self.ts_plot_feats.options = list(df.columns)
        self.pc_plot_feats.options = list(df.columns)
        self.kde_plot_feats.options = list(df.columns)
                                
        self.wb_slider.start = 0
        self.wb_slider.step  = 1
        self.wb_slider.end   = len(df)-1
        self.wb_slider.value = (self.wb_slider.start, self.wb_slider.end)
        
        ### Tigger a callback if, also if the column 
        self.ts_plot_feats.value = []
        self.pc_plot_feats.value = []
        self.kde_plot_feats.value = []
        ###
        
        num_preview_columns = min(4, len(df.columns))
        preview_columns = [col_ for col_ in df.columns[0:num_preview_columns]]
        
        matches = sum([feat_ in preview_columns for feat_ in self.ts_plot_feats.value])
        if matches != 0:
            self.ts_plot.object = get_empty_ts_plot_figure() # Reset the trace data of the ts_plot. If having traces with the same name, they wont get updated
        self.ts_plot_feats.value = preview_columns
        self.pc_plot_feats.value = preview_columns
        self.kde_plot_feats.value = preview_columns
        self.update_ts_plot_features(event='Direct Call') # calling these plotting callbacks a second time to rerender the figure
        self.update_kde_plot(event='Direct Call')         #  (solves rendering bug, after clearing all data; plot and legend overlapping)
         
    def correct_color_faeture_selection(self):
        """ Corrects the options for color features upon changing the selection of paralell coordinate features.
            - The color feature has to be inside the paralell coordinate features.
        """
        self.pc_coloring_feats.options = self.pc_plot_feats.value
        if (not self.pc_coloring_feats.value in self.pc_plot_feats.value) and (self.pc_plot_feats.value != []):
            self.pc_coloring_feats.value = self.pc_plot_feats.value[0]
            
    def update_syncronized_feature_selection(self, event):
        """ This function should update the synchronized feature selection, when changing the checkbox value to True. 
            This will trigger multiple plotting callbacks! 
        """
        if   self.tabs.active == 0:
            self.pc_plot_feats.value  = self.ts_plot_feats.value 
            self.kde_plot_feats.value = self.ts_plot_feats.value
        elif self.tabs.active == 1:
            self.ts_plot_feats.value  = self.pc_plot_feats.value 
            self.kde_plot_feats.value = self.pc_plot_feats.value
        elif self.tabs.active == 2:
            self.ts_plot_feats.value  = self.kde_plot_feats.value 
            self.pc_plot_feats.value  = self.kde_plot_feats.value
            
    def range_slider_callback(self, event):
        """ Only update the visible plot for speed and easier programming. """
        if self.tabs.active == 0: 
            self.update_ts_plot_index_range()
        if self.tabs.active == 1: 
            self.update_pc_plot(event='range_slider_callback')
        if self.tabs.active == 2: 
            self.update_kde_plot(event='range_slider_callback')
    
    def range_slider_callback_throttled(self, event):
        """ Only update the visible plot when updates are too expensive (not the case) """
        if self.tabs.active == 0: 
            pass 
        if self.tabs.active == 1: 
            pass 
        if self.tabs.active == 2: 
            pass 
        
    def update_feat_scaling(self, event):
        if self.tabs.active == 0: 
            self.update_ts_plot_features(event='Feature Scaling Callback') 
        if self.tabs.active == 1: 
            pass  
        if self.tabs.active == 2: 
            self.update_kde_plot(event='Feature Scaling Callback') 
    
    ### Time Series Plot Callbacks
    def update_ts_plot_features(self, event):
        if event == 'Feature Scaling Callback':
            did_scaling_method_change = True
        else:
            did_scaling_method_change = False
        self.ts_plot.object = update_ts_plot_figure(fig=self.ts_plot.object, 
                                                    df=self.env_data, 
                                                    col_names=self.ts_plot_feats.value,
                                                    scaling_method=self.feat_scaling_method.value,
                                                    did_scaling_method_change=did_scaling_method_change)
        
    def update_ts_plot_index_range(self):
        update_ts_plot_index_range(fig=self.ts_plot.object, 
                                   idx_range=self.wb_slider.value, 
                                   df_index=self.env_data.index)
    ###
    
    ### Parallel Coordinate Callbacks
    def update_pc_plot(self, event):
        self.correct_color_faeture_selection() 
        if self.pc_coloring_feats.value != None:
            self.pc_plot.object = update_pc_plot(df=self.env_data, 
                                                 idx_range=self.wb_slider.value,
                                                 col_names=self.pc_plot_feats.value,
                                                 color_column=self.pc_coloring_feats.value)
    ###
    
    
            
    ### KDE Plot Callbacks/Functions
    def get_non_unit_variance_kde_features(self, idx_range): # default idx_range indexes the entire df
        removed_feats = []
        for feat_ in self.kde_plot_feats.value:
            if self.env_data.iloc[idx_range[0]:idx_range[1]+1][feat_].var() == 0: # cant do KDE with a flatline features; removing them
                removed_feats += [feat_]
        kept_feats = [feat_ for feat_ in self.kde_plot_feats.value if feat_ not in removed_feats]
        return kept_feats
    
    def update_kde_plot(self, event):
        """ A collective callback when changing the kde features, index range, bandwidth, or scaling method """
        non_unit_variance_feats = self.get_non_unit_variance_kde_features(idx_range=self.wb_slider.value)
        self.kde_plot.object = update_kde_plot(df=self.env_data, 
                                               idx_range=self.wb_slider.value, 
                                               col_names=non_unit_variance_feats,
                                               bw_adjust=self.band_width_slider.value,
                                               scaling_method=self.feat_scaling_method.value)
    ###
      
    def save_figure(self, event):
        """ The figures are saved in SVG format, as the plotly png writers are very bad. 
            The images all look unreadable and out of proportion.
        """
        if self.tabs.active == 0: 
            file_prefix = 'ts_plot_'
            file_name = find_new_numeric_file_name(base_dir=self.exp_save_dir, file_prefix=file_prefix, file_extension='.svg') 
            self.ts_plot.object.write_image(self.exp_save_dir + '\\' + file_name, scale=1)
        if self.tabs.active == 1: 
            file_prefix = 'pc_plot_'
            file_name = find_new_numeric_file_name(base_dir=self.exp_save_dir, file_prefix=file_prefix, file_extension='.svg') 
            self.pc_plot.object.write_image(self.exp_save_dir + '\\' + file_name, scale=1)
        if self.tabs.active == 2: 
            file_prefix = 'kde_plot_'
            file_name = find_new_numeric_file_name(base_dir=self.exp_save_dir, file_prefix=file_prefix, file_extension='.svg') 
            self.kde_plot.object.write_image(self.exp_save_dir + '\\' + file_name, scale=1)
        
        
""" Note that pn.bind is NEVER used, even if it is recommended for multi-input callback functions (like several plotting callbacks here):
        - I do not like, that the pn.bind is directly giving parameter values to the callback function
        - In this class-based UI, I prefer to extract the widget states from the self argument, which is a bit clearer coding-wise
            --> But with the pn.bind, you do not see from which component which values initially comes from
            --> especially, when doing component value corrections like rounding, etc. you want to know that a value is only specified to a single component
        - Multiple .param.watch() callbacks do fine here, and also offer way more flexibility! --> see range_slider callbacks
        - I also prefer to directly specify each callback for each widget in strictly separate lines!
"""