import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt

### Matplotlib ###
def reset_matplotlib_default_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize'        : [6.4, 4.8],
                        #'axes.prop_cycle'       : plt.cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']),
                        'axes.prop_cycle'       : plt.cycler('color', ['#D03030','#7070DD','#30C030','#E09020','#30A0A0','#8060CC','#DD60C0']),
                        'lines.linewidth'       : 1,
                        'lines.markersize'      : 4,
                        'axes.labelsize'        : 11,
                        'xtick.labelsize'       : 10,
                        'ytick.labelsize'       : 10,
                        'xtick.minor.visible'   : True,
                        'xtick.top'             : True,
                        'xtick.direction'       : 'in',
                        'ytick.minor.visible'   : True,
                        'ytick.right'           : True,
                        'ytick.direction'       : 'in',
                        'savefig.bbox'          : 'tight',
                        'legend.frameon'        : False,
                        'grid.linewidth'        : 0.5,
                        'savefig.pad_inches'    : 0.05,
                        'font.family'           : 'serif',
                        'mathtext.fontset'      : 'dejavuserif',
                        'text.usetex'           : False,
                        'animation.embed_limit' : 1000.0}) # 1 GB

### Plotly ###
def get_empty_general_figure(title, xaxis_title, yaxis_title, legend_title):
    """ Gets emtpy template figures for all general line plots, scatter plots, ... in Plotly.
        The format should look similar to matplotlib, but highly polished and scientific.
    """
    fig = px.line(pd.DataFrame(), width=800, height=500, title=title,
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_xaxes(gridwidth=0.5, showgrid=True, gridcolor='rgba(150,150,150,0.4)',                     # x-major grid
                        zeroline=False,                                                                   # misc
                        showline=True, linewidth=1, linecolor='black', mirror='ticks',                    # x-border
                        nticks=10, ticks="inside", ticklen=4.5, tickwidth=1.4,                            # major x-ticks
                        minor=dict(ticks="inside", ticklen=2.5, tickwidth=1.0, showgrid=False, nticks=6), # minor x-ticks
                        tickfont=dict(family='Serif', size=16),                                           # x-tick labels
                        title=dict(text=xaxis_title, standoff=14, font=dict(family='Serif', size=18)), # LaTeX font not affected by fontsize
                    )

    fig.update_yaxes(gridwidth=0.45, showgrid=True, gridcolor='rgba(150,150,150,0.4)',                    # x-major grid
                        zeroline=False,                                                                   # misc
                        showline=True, linewidth=1, linecolor='black', mirror='ticks',                    # x-border
                        nticks=10, ticks="inside", ticklen=4.5, tickwidth=1.4,                            # major x-ticks
                        minor=dict(ticks="inside", ticklen=2.5, tickwidth=1.0, showgrid=False, nticks=5), # minor x-ticks
                        tickfont=dict(family='Serif', size=15),                                           # x-tick labels
                        title=dict(text=yaxis_title, standoff=14, font=dict(family='Serif', size=18)), # LaTeX font not affected by fontsize
                    )

    fig.update_layout(title=dict(font=dict(family='Serif', size=21)), #'DejaVu Serif'
                        title_y=1.0, title_x=0.5, title_xanchor='center', title_xref='paper',
                        legend=dict(title=dict(text=legend_title, font=dict(family='Serif', size=18)),
                                    font=dict(family='Serif', size=15)),
                        plot_bgcolor = 'white',
                        showlegend=True,
                        margin=dict(t=30,l=30,r=0,b=30),
                        modebar=dict(remove=['select', 'zoomIn', 'zoomOut', 'resetScale', 'toImage'])) # These should never be needed, they are useless
    return fig

def get_empty_square_animation_fig(title, xaxis_title, yaxis_title, x_lims=(-1,1), y_lims=(-1,1), margin=dict(t=30,l=30, r=30, b=30)):
    """ Returns an empty figure with equal x-axis and y-axis scaling that looks square.
        This is often used for animating physics simulations.

        a margin like margin=dict(t=30,l=30, r=30, b=30) should be set to get equal margins, since it affecs the scaleratio.
        The scaleratio needs to be the same on x and y to get a square figure. Adjust the margin for animations to maintain it when adding buttons, etc.
    """
    fig = get_empty_general_figure(title=title,
                                   xaxis_title=xaxis_title,
                                   yaxis_title=yaxis_title,
                                   legend_title=None)

    fig.update_layout(
        # Make a square figure with equally scaled axes:
        margin=margin,
        yaxis=dict(scaleanchor="y", scaleratio=1, range=[x_lims[0], x_lims[1]], nticks=12, title=dict(standoff=0)), # also sets 12 instead of 10 ticks
        xaxis=dict(scaleanchor="x", scaleratio=1, range=[y_lims[0], y_lims[1]], nticks=12, title=dict(standoff=10)),

        title_y=1.0,
        showlegend=False,
        dragmode=None, #'pan', 'zoom',
        modebar=dict(remove=['select', 'autoscale', 'toImage', 'lasso2d']),
        width=700,
        height=700,
        autosize=False,
    )

    return fig

def get_fig_anim_widgets(fig, frame_freq, transition_duration=0, slider_x=0.084, player_x=0.84):
        """ This function will return video player widget for an animation figure. The figure must have fig.frames set!
            The widgets are specialized for physical simulation animations.
            frame_freq : the frequency of animation; = 1/fps; this should be the same frequency in which the frames are samples from a simulation time series.
        """
        assert (fig.frames != ()), ('The fig must have its animation frames set in order to properly initialize the slider.')

        def frame_args(duration, direction='forward'):
            """ Parameters you can use: https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js """
            return dict(frame=dict(duration=duration, redraw=False),
                        mode='immediate',
                        direction=direction, # 'forward', 'reverse'
                        fromcurrent=True,
                        transition=dict(duration=transition_duration, easing='cubic')) # 'cubic', 'linear'; transition duration should be defined for regular data; but it causes flickering for physics simulations

        def get_slider():
            slider = dict(pad          = {'b' : 10, 't': 58},
                          len          = 0.8,
                          x            = slider_x, #0.093,
                          y            = 0,
                          active       = 0,
                          visible      = True,
                          minorticklen = 0,
                          ticklen      = 0, # looks simpler without ticks and labels!
                          currentvalue =  dict(prefix = '', suffix='', visible = False),
                          steps        = [dict(args  = [[frame_.name], frame_args(0)],
                                               label = '', # f"{float(frame_.name.replace('frame', '')):.{num_decimals}f}" # 'frame10.15' --> '10.150' ; removed labels for visual simplicity
                                               method = 'animate',
                                              ) for frame_ in fig.frames] )
            return slider

        def get_video_player_buttons():
            button_slowest = dict(args   = [None, frame_args(frame_freq*1000*10)],
                                  label  = '×0.1', # × '&#xD7;'
                                  method = 'animate')
            button_slower  = dict(args   = [None, frame_args(frame_freq*1000*4)],
                                  label  = '×0.25',
                                  method = 'animate')
            button_slow    = dict(args   = [None, frame_args(frame_freq*1000*2)],
                                  label  = '×0.5',
                                  method = 'animate')
            button_step_l  = dict(args   = [None, frame_args(frame_freq*1000*10000, direction='reverse')], # this actually works as a frame step per click! (but not when it is the first or last frame); this but still very useful
                                  label  = '⏪︎',  # ⏪︎ &#x23EA;
                                  method = 'animate')
            button_reverse = dict(args   = [None, frame_args(frame_freq*1000, direction='reverse')],
                                  label  = '◀',  # ◀ &#x25C0; ⏴ &#x23F4;
                                  method = 'animate')
            button_pause   = dict(args   = [[], frame_args(None)], # by specifying [] the the animation stops when calling 'animate'
                                  label  = '■',  # ⏸ &#x23F8; ⏹ &#x23F9; ■ &#x25FC; ◾ &#x25FE; ▪ &#x25AA;
                                  method = 'animate')
            button_play    = dict(args   = [None, frame_args(frame_freq*1000, direction='forward')],
                                  label  = '▶',  # ▶ &#x25B6; ⏵ &#x23F5;
                                  method = 'animate')
            button_step_r  = dict(args   = [None, frame_args(frame_freq*1000*10000, direction='forward')],
                                  label  = '⏩︎',  # ⏩︎ &#x23E9;
                                  method = 'animate')
            button_fast    = dict(args   = [None, frame_args(np.round(frame_freq*1000/1.25))],
                                  label  = '×1.25',
                                  method = 'animate')
            button_faster  = dict(args   = [None, frame_args(np.round(frame_freq*1000/1.5))],
                                  label  = '×1.5',
                                  method = 'animate')
            button_fastest = dict(args   = [None, frame_args(frame_freq*1000/2.0)], # often x2.0 can already be the max; plotly renders each frame without skipping
                                  label  = '×2.0',
                                  method = 'animate')
            buttons = [button_slowest, button_slower, button_slow,
                       button_step_l, button_reverse,
                       button_pause,
                       button_play, button_step_r,
                       button_fast, button_faster, button_fastest]

            return buttons



        def get_video_player_menu():
            video_player_menu = dict(borderwidth = 1.4,
                                     showactive  = True,
                                           font  = dict(size= 14, family='Segoe UI Symbol'), # These font families prevent unicode being rendered as an emoji!
                                               x = player_x, #0.93,
                                               y = 0,
                                             pad = {'r': 10, 't': 82},
                                            type = 'buttons',
                                       direction = 'right',
                                        buttons  = get_video_player_buttons())
            return video_player_menu



        slider     = get_slider()
        updatemenu = get_video_player_menu()
        return updatemenu, slider

def get_fig_style_menu(fig, tracename2id_mapping : dict, toggled_trace_names : list, dropdown_pos =(1.3,1.0), fig_margin=dict(t=30,l=120, r=30, b=30)):
    """ This function can returns all sorts of styling menus when implemented.
        But for now it just returns a style dropdown to enable/disable certain traces of a figure.
    """
    assert (np.prod([trace_name in tracename2id_mapping.keys() for trace_name in toggled_trace_names]) == 1), \
           (f"All toggled_trace_names {toggled_trace_names} should also be in tracename2id_mapping {tracename2id_mapping}!")

    def get_styling_buttons():
        """ These buttons are to toggle specific traces which helps to focus on specific information. """
        tn2id_ = tracename2id_mapping # renaming

        buttons_toggle = [dict(args   = [{'visible' : False},  # trace parameters
                                         {},                   # style parameters
                                         tn2id_[label_]        # ids of affected traces (has to be list)
                                        ],
                               args2  = [{'visible' : True},   # args2 get called upon clicking an active button (enables toggling)
                                         {},
                                         tn2id_[label_]
                                        ],
                               label  = label_,
                               method = 'update')
                          for label_ in toggled_trace_names]

        buttons = buttons_toggle # possible to add other buttons here too
        return buttons

    def get_styling_menu():
        styling_menu = dict(borderwidth  = 1.4,
                             showactive  = True,
                                   font  = dict(size= 14, family='Segoe UI Symbol'),
                                       x = dropdown_pos[0],
                                       y = dropdown_pos[1],
                                    type = 'dropdown',
                               direction = 'down',
                                buttons  = get_styling_buttons())
        return styling_menu

    updatemenu = get_styling_menu()
    return updatemenu

def update_figure_with_anim_widgets_and_style_buttons(fig,
                                                      frame_freq,
                                                      tracename2id_mapping,      # dict with format {'trace_group_name' : [1,2,3]} # [1,2,3] are indizes of traces belonging to 'trace_group_name'
                                                      toggled_trace_names : list, # these have to be keys in tracename2id_mapping
                                                      transition_duration=0,
                                                      slider_x=0.084,
                                                      player_x=0.84,
                                                      dropdown_pos =(1.3,1.0),
                                                      fig_margin=dict(t=30,l=120, r=30, b=30)):
    """ A function which both adds an animation player and style buttons on the right.
        This is useful for defining everything in one simple function call (faster trial-and-error set up of the figure).
    """
    anim_menu, anim_slider = get_fig_anim_widgets(fig, frame_freq, transition_duration=transition_duration, slider_x=slider_x, player_x=player_x)
    style_menu             = get_fig_style_menu(fig, tracename2id_mapping=tracename2id_mapping, toggled_trace_names=toggled_trace_names, dropdown_pos=dropdown_pos)
    fig.update_layout(updatemenus=[anim_menu, style_menu], sliders=[anim_slider], margin=fig_margin)

