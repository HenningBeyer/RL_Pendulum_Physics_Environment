import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objs as go
from scipy.stats import gaussian_kde
from rl_pendulum.utils.plotting import get_empty_general_figure
from sklearn.preprocessing import MinMaxScaler, StandardScaler



### Time Series Plot Functions
def get_empty_ts_plot_figure():
    """ Returns only the figure.
        --> The figure layout was separated from the data assignment for performance.
    """
    fig = get_empty_general_figure(title='Time Series Feature Visualization',
                                   xaxis_title=r"$\large{t \text{ (s)}}$",
                                   yaxis_title=r'$\large{\text{Features (a.u.)}}$',
                                   legend_title=None, #'Features:', #This misaligns when using latex strings as features
                                   )
    return fig

def update_ts_plot_figure(fig, df, col_names, scaling_method, did_scaling_method_change=False):
    """ Columns are column names of the df which should be plotted

       Re-rendering the entire figure might not be as performant, as just updating altered traces.
       Plotly only allows to assign a subset of all previous traces to fig.data; assigning new traces is not possible --> fig.add_traces
    """
    # always fit the scaler to the complete data (not affected by index_range)
    if len(col_names) > 0:
        df_idx_ = df.index
        if scaling_method == 'Standard Scaling':
            scaler = StandardScaler()
            df = pd.DataFrame(scaler.fit_transform(df[col_names]), columns=col_names)
            df.index = df_idx_
        elif scaling_method == 'Min-Max Scaling':
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df[col_names]), columns=col_names)
            df.index = df_idx_
        elif scaling_method == 'No Scaling':
            pass

    # Remove traces:
    # (removes traces not present in col_names anymore)
    if did_scaling_method_change:
        kept_traces = []
        fig = get_empty_ts_plot_figure() # get a new figure to prevent a graphic bug (legend and plot overlap)
    else:
        kept_traces = [trace_ for trace_ in fig.data if trace_.name in col_names]
        fig.data = kept_traces

    # Add new traces
    # (only adds new traces not already present)
    kept_traces_names = [trace_.name for trace_ in kept_traces]
    new_trace_names = [name_ for name_ in col_names if name_ not in kept_traces_names]
    new_data = list([go.Scatter(x=df.index, y=df[trace_name_], name=trace_name_, mode='lines') for trace_name_ in new_trace_names])
    fig.add_traces(new_data)
    return fig

def update_ts_plot_index_range(fig, idx_range, df_index):
    """ Instead of slicing the data, only the plot x_range is altered for performance. """
    x1 = df_index[idx_range[0]] # idx_range (50,1050) for example
    x2 = df_index[idx_range[1]]
    fig.update_xaxes(range=(x1,x2)) # x_range (1.5,10.5) or ('25.04.2024 00:00', '25.04.2024 12:04') for example
###

### Parallel Coordinate Functions
def get_empty_pc_figure():
    """ Updating a parallel coordinate plot without redrawing is not possible.
        If altering any parameter of a parallel coordinate figure, the plot has to be redrawn, so this function has very limited use.

        The function just makes nice empty placeholder parallel coordinate plot with title and 3 dragable axes with ticks.
        This would only be used if no data at all is present.
    """
    # making a nice empty placeholder parallel coordinate plot (has title + 3 dragable axes with ticks)
    fig = px.parallel_coordinates(data_frame=pd.DataFrame([[0,0,0]], columns=['', ' ', '  ']), color=[0], width=800, height=500,
                                    title='Parallel Coordinate Feature Analysis')
    fig.update_layout(title=dict(font=dict(family='Serif', size=20)),
                        title_y=1.0, title_x=0.5, title_xanchor='center', title_xref='paper',
                        coloraxis_showscale=True,
                        margin=dict(t=65,l=30,r=10,b=20),
                        modebar=dict(remove=['toImage']))
    return fig

def update_pc_plot(df, col_names, color_column, idx_range):
    """ The complete figure has to be reset to avoid weird update flickering. This is still very fast, surprisingly.
        If features are set too fast, a flicker loop would happen.
    """
    fig = px.parallel_coordinates(data_frame=df.iloc[idx_range[0]:idx_range[1]+1][col_names].reset_index(drop=True),  #reset_index important: else a lot of nan values
                                  color=color_column,
                                  width=800, height=500,
                                  title='Parallel Coordinate Feature Analysis')
    fig.update_layout(title=dict(font=dict(family='Serif', size=20)),
                        title_y=1.0, title_x=0.5, title_xanchor='center', title_xref='paper',
                        coloraxis_showscale=True,
                        margin=dict(t=65,l=30,r=10,b=20),
                        modebar=dict(remove=['toImage']))
    return fig
###

### KDE Plot Functions
def sns_kdeplot_data(df, cut=3, gridsize=500, bw_method='scott', bw_adjust=1.0):
    """ This is a replicate of seaborn.kdeplot(). (the results matched exactly on testing with data)
        Seaborn was very buggy, as it often overwrote plt and plotly figures, and caused weird caching in VSCode.
        For example seaborn.kdeplot() (with no arguments!) would yield the same as seaborn.kdeplot(data=df) in another function where df was not even defined!.

        But the code below works like sns.kdeplot, and gives the exact results seaborn would yield, without the bugs.
        It does return only the data; plotting should be done with other libraries.
        It is also specialized to only work on a pd.DataFrame with numeric values.
    """

    cut_factor = 1+cut/100 # 3 --> 1.03
    x_data = np.array([np.linspace(df[col_].min()*cut_factor,
                                df[col_].max()*cut_factor,
                                num=gridsize) for col_ in df.columns])

    kde_insts = [gaussian_kde(df[col_].dropna(), bw_method=bw_method) for col_ in df.columns]
    for kde_ in kde_insts:
        kde_.set_bandwidth(kde_.factor*bw_adjust)

    y_data = np.array([kde_(x_data_) for kde_, x_data_ in zip(kde_insts, x_data)])

    return x_data, y_data # these variables contrain one trace for each column and need to be unpacked later


def get_empty_kde_plot_figure(one_feature=False):
    """ Returns only the figure.
        --> The figure layout was separated from the data assignment for performance.
    """
    fig = get_empty_general_figure(title='KDE Feature Distributions',
                                   xaxis_title=r"$\large{\text{Features (a.u.)}}$",
                                   yaxis_title=r'$\large{\text{Density}}$',
                                   legend_title=None, #'Features:', #This misaligns when using latex strings as features
                                   )
    return fig

def update_kde_plot(df, col_names, idx_range, bw_adjust, scaling_method):
    """ This is a function for updating all possible changes of parameters in one function.
        This is no problem, as the data has to be recalculated for any change.
    """


    if scaling_method not in ['Standard Scaling', 'Min-Max Scaling', 'No Scaling']:
        raise ValueError(f'A wrong scaling_method {scaling_method} was specified for KDE plotting.')

    # always fit the scaler to the complete data (not affected by index_range)
    if len(col_names) > 0:
        if scaling_method == 'Standard Scaling':
            scaler = StandardScaler()
            df = pd.DataFrame(scaler.fit_transform(df[col_names]), columns=col_names)
        elif scaling_method == 'Min-Max Scaling':
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df[col_names]), columns=col_names)
        elif scaling_method == 'No Scaling':
            pass

    fig = get_empty_kde_plot_figure(one_feature=(len(col_names) == 0))
    # It is very flickery, when doing fig.data = [] for a line plot; so a complete reassignment is neccessary
    x_data, y_data = sns_kdeplot_data(df=df.iloc[idx_range[0]:idx_range[1]+1][col_names], cut=0.0, gridsize=500, bw_adjust=bw_adjust)
    new_data = [go.Scatter(x=x_data[i], y=y_data[i], mode='lines', name=col_names[i]) for i in range(len(col_names))]
    fig.add_traces(new_data)
    return fig
###


