import panel as pn
import numpy as np
import os

from rl_pendulum.ui.styling import wb_h2_header_margins, divider_margins, wb_h1_header_margins, main_wb_margins, sub_wb_margins
from rl_pendulum.utils.file_utils import find_new_numeric_dir_name

class Home_UI():
    def __init__(self):
        desc_mapping = {'new_exp'             : "• Start a new experiment with the standard parameters. ",  # buttons dont support line-break descriptions
                        'load_exp'            : "• Load a previous experiment with its data mainly to re-visualize the results again \n" + \
                                                "• It is also possible to reconduct previous experiments with this.",
                        'exp_save_dir'        : "- Chose a directory to save your experiments in. \n " + \
                                                "- It is intended to save experiments inside './experiments' \n " + \
                                                "- Consider storing experiments in group directories for more clarity (create them by hand).",
                        'exp_name'            : "- Name your experiment directory for faster identification.",    
                        'exp_load_dir'        : "- Specify the experiment directory being loaded as a new duplicated experiment \n " +\
                                                "- Only used for the button 'Load Experiment Data'. \n" +\
                                                "- Specify 'Experiment Save Directory' and 'Experiment Name' above, as their values will be used."
                        }
        
        self.base_exp_dir = os.getcwd()+'\\experiments'
        self.base_exp_prefix = 'example_experiment_'
        self.exp_name = None
            
        self.button_start = pn.widgets.Button(name="New Experiment", description=desc_mapping['new_exp'])
        self.button_load  = pn.widgets.Button(name="Load Experiment Data", description=desc_mapping['load_exp'])
        
        self.exp_save_dir_input   = pn.widgets.TextInput( width=500, name='Experiment Save Directory', value= self.base_exp_dir, description=desc_mapping['exp_save_dir'])
        self.exp_name_input       = pn.widgets.TextInput( width=500, name='Experiment Name', description=desc_mapping['exp_name'],
                                                    value=find_new_numeric_dir_name(base_dir=self.base_exp_dir, dir_prefix=self.base_exp_prefix))
        self.exp_load_dir_input   = pn.widgets.TextInput( width=500, name='Experiment Load Directory', value= os.getcwd()+'\\experiments\\experiment_1', description=desc_mapping['exp_load_dir'])
    
    
        self.start_experiment_wb = pn.layout.WidgetBox(
            pn.pane.Markdown('## Start Experimenting', margin=wb_h2_header_margins),
            pn.layout.Divider( margin=(0,0,0,0)),
            self.button_start, 
            self.button_load,
            margin=sub_wb_margins, styles={'border-width' : '2.0px'}
        )
        self.experiment_setting_wb = pn.layout.WidgetBox(
            pn.pane.Markdown('## Experiment Settings', margin=wb_h2_header_margins),
            pn.layout.Divider(),
            self.exp_save_dir_input,
            self.exp_name_input,
            self.exp_load_dir_input,
            margin=sub_wb_margins, min_width=500, styles={'border-width' : '2.0px'}
        )
        self.ui = pn.layout.WidgetBox(
                    *[pn.pane.Markdown('# Home Menu', margin=wb_h1_header_margins),
                      pn.Row(self.start_experiment_wb, self.experiment_setting_wb)],
                    margin=main_wb_margins, styles={'border-width' : '2.0px'}, min_height=300 # weird behaviour: have to specify min_height with any value, so that the widgetbox of self.ui stretches automatically
                    )
        
    def prepare_new_experiment_dir(self):
        """ Creates a complete experiment directory structure"""
        exp_dir = self.exp_save_dir_input.value + '\\' + self.exp_name
        new_dirs = ['benchmark', 'model', 'monitoring', 'params', 'saved data']
        for dir_ in new_dirs:
            os.mkdir(exp_dir + '\\' + dir_)
        created_dirs = {dir_ : exp_dir + '\\' + dir_ for dir_ in new_dirs}
        return created_dirs # this return is for utility
        
    def set_new_exp_name(self):  
        """ update the experiment name already after starting a new experiment; enables to press 'New Experiment' again """
        self.exp_name = self.exp_name_input.value
        self.exp_name_input.value = find_new_numeric_dir_name(base_dir=self.exp_save_dir_input.value, dir_prefix=self.base_exp_prefix)