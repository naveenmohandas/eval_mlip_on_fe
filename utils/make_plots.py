from collections.abc import Sequence
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm


sns.set_style("whitegrid", {'axes.grid' : False})
print("setting sns.set_tyle to whitegrid")

plt.rcParams['svg.fonttype'] = 'none'
print("setting svg.fonttype to none")

# for dark background
# sns.set(style="ticks", context="talk")
# plt.style.use("dark_background")


SAVE_PLOTS = False
print(f"setting SAVE_PLOTS: {SAVE_PLOTS}")


def get_rmse(x, y):
    """
    Get two list of values and determine 
    the rmse
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    assert len(x) == len(y), "Length mismatch for x and y"

    squared_error = (x - y)**2
    average =  sum(squared_error) / len(x)

    return np.sqrt(average)

#  make subplots


def create_hist2d_subplots_with_single_cmap(dft_energy_list: Sequence[list], predicted_energy_list: Sequence[list], 
                          model_names: Sequence[str], 
                          xmin: float, xmax: float, rmses: list,
                          xlabel: str, ylabel:str,
                          colocsceheme: str = 'Blues',
                          cols = None, rows=1,
                          binsize=None, 
                          save_dir=None,
                          save_name:str='',
                          save_single:bool=False,
                          add_fig_numbering:bool=True):
    """
    It createes a 2D histogram subplots for each model
    Parameters
    ----------
    dft_energy_list : Sequence[list]
        List of DFT energy values for each model.
    predicted_energy_list : Sequence[list]
        List of predicted energy values for each model.
    model_names : Sequence[str]
        List of model names corresponding to the energy lists.
    xmin : float
        Minimum x-axis value for the histogram.
    xmax : float
        Maximum x-axis value for the histogram.
    rmses : list
        List of RMSE values for each model.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    colocsceheme : str, optional
        Colormap scheme for the histogram. Default is 'Blues'.
    cols : int, optional
        Number of columns for the subplot grid. If None, it will be calculated.
    rows : int, optional
        Number of rows for the subplot grid. If None, it will be calculated.
    binsize : list, optional
        Size of the bins for the histogram. If None, default is [40, 40].
    save_dir : Path | None, optional
        Directory to save the plots. If None, plots will not be saved.
    save_name : str, optional
        Name to save the plots. Default is an empty string.
    save_single : bool, optional
        If True, saves each subplot as a separate image. Default is False.


    """
    # Number of models
    n = len(model_names)

    if binsize is None:
        binsize = [40,40]
    
    # Determine the grid layout: roughly square, adjust rows and cols
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                            #  sharey="row")
    
    # Flatten axes array for easy iteration (handles both 1D and 2D cases)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Collect histogram counts to determine common normalization
    all_counts = []
    for i in range(n):
        dft_energy = dft_energy_list[i]
        predicted_energy = predicted_energy_list[i]
        counts, _, _ = np.histogram2d(dft_energy, predicted_energy, bins=binsize)
        all_counts.append(counts)
    
    # Determine global min and max counts (excluding empty bins)
    max_count = max(c.max() for c in all_counts)
    min_count = min(c[c > 0].min() for c in all_counts) if any(c[c > 0].size for c in all_counts) else 1
    
    # Create logarithmic normalization
    norm = LogNorm(vmin=min_count, vmax=max_count)
    
    # tick_values = np.linspace(int(xmin), int(xmax), 6)
    tick_values = np.linspace(xmin, xmax, 4)
    tick_labels = [f"{val:.1f}" for val in tick_values]
    print(f"{tick_labels = }{tick_values = }")
    # Iterate over each model and create a hist2d plot
    for i in range(n):
        dft_energy = dft_energy_list[i]
        predicted_energy = predicted_energy_list[i]
        model_name = model_names[i]
        rmse = rmses[i]
        
        ax = axes[i]
        # Hist2D plot with Blues colormap and shared normalization
        hist = ax.hist2d(dft_energy, predicted_energy, bins=binsize, 
                        cmap=colocsceheme, norm=norm)
        
        # Plot y=x line
        ax.plot([xmin, xmax], [xmin, xmax], '--k', alpha=0.7)
        
        # Set title and labels
        ax.set_title("" + model_name +f". (RMSE: {rmse:.3f})", fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.tick_params(axis='both', which='major', labelsize=8)

        ax.set_xticks(tick_values, tick_labels, fontsize=12)
        ax.set_yticks(tick_values, tick_labels, fontsize=12)
        if add_fig_numbering:
            ax.text(-0.15, 1.01, f"{chr(97+i)})", transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='bottom', ha='right')

        if save_single:
            # Save individual subplot in a separate figure to avoid overlap
            temp_fig = plt.figure(figsize=(5, 4))
            temp_ax = temp_fig.add_subplot(111)
            temp_hist = temp_ax.hist2d(dft_energy, predicted_energy, bins=[40, 40], 
                                    cmap='Blues', norm=norm)
            temp_ax.plot([xmin, xmax], [xmin, xmax], '--k', alpha=0.7)
            temp_ax.set_title(f"{model_name}. RMSE: {rmse:.3f}", fontsize=10)
            temp_ax.set_ylabel("MLIP predicted energy eV/atom", fontsize=8)
            temp_ax.set_xlabel("DFT predicted energy eV/atom", fontsize=8)
            temp_ax.set_xlim(xmin, xmax)
            temp_ax.set_ylim(xmin, xmax)
            temp_ax.tick_params(axis='both', which='major', labelsize=12)
            temp_fig.tight_layout()
            temp_fig.savefig(f"{save_dir}/subplot_{model_name}.png", dpi=500, bbox_inches='tight')
            plt.close(temp_fig)
    
    
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # fig.supxlabel(xlabel, fontsize=14)
    # fig.supylabel(ylabel, fontsize=14)
    # axes[0].set_ylabel(ylabel, fontsize=14)

    # Adjust layout to make space for the colorbar on the right
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make space for the colorbar
    # plt.subplots_adjust(left=0.1)  # Make space for the ylabel
    # plt.subplots_adjust(bottom=0.1)  # Make space for the xlabel
    
    # Add a single colorbar for the entire figure, positioned outside
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(hist[3], cax=cbar_ax)
    cbar.set_label('Count (log scale)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    for _, spine in cbar.ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)


    ## below snippet added for borders
    for ax in axes:
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
     
    if save_single:
        # Save the colormap as a separate image
        fig_cbar = plt.figure(figsize=(1, 4))
        cbar_ax = fig_cbar.add_axes([0.1, 0.1, 0.3, 0.8])
        fig_cbar.colorbar(hist[3], cax=cbar_ax)
        cbar_ax.set_ylabel('Count (log scale)', fontsize=10)
        cbar_ax.tick_params(labelsize=8)
        fig_cbar.savefig(f"{save_dir}/colormap.png", dpi=500, bbox_inches='tight')
        plt.close(fig_cbar)

    # Save the figure
    if save_dir:
        save_path = Path(f"{save_dir}/{save_name.strip('pkl')}")
        print("saving hist2d subplot in ", save_path)
        plt.savefig(f"{save_path}_subplots.png", dpi=800, bbox_inches='tight', transparent=True)
        plt.savefig(f"{save_path}_subplots.svg",bbox_inches='tight')

    else:
        print("not saving hist2d subplot")
        print("Set SAVE_PLOTS to True to save the plots")
    
    return fig


def plot_fe_energies(plot_info:dict, experiments:list, remove_for_fe:bool, plot:str,
                     base_dir:Path, file_name:str, save_dir: Path|None ):

    mlip_key = plot_info[plot]['mlip_key']
    dft_key = plot_info[plot]['dft_key']

    print(f"{mlip_key = }, {dft_key = }")
    data=None

    dft_energies, predicted_energies, rmses = [], [], []
    for exp in experiments:
        data_path = base_dir / exp / file_name    
        data = None
        with open(data_path, 'rb') as f:
            data_raw = pickle.load(f)
        plot_data = {dft_key:data_raw[dft_key] ,
                     mlip_key:data_raw[mlip_key], }
        data = pd.DataFrame(plot_data)
        min_dft = data[dft_key].min()
        max_dft = data[dft_key].max()
        rmse = get_rmse(data[dft_key], data[mlip_key])
        if remove_for_fe:
            data = data[data[dft_key]< -6.]

        dft_energies.append(data[dft_key].values)
        predicted_energies.append(data[mlip_key].values)
        rmses.append(rmse)

    xmin = -8.4       # based on the lowest dft value
    xmax = max(data[dft_key])
    print(f"{xmin = }, {xmax = }")

    fig = create_hist2d_subplots_with_single_cmap(
        dft_energy_list=dft_energies,
        predicted_energy_list=predicted_energies,
        model_names=experiments,
        xmin=xmin,
        xmax=xmax,
        rmses= rmses,
        xlabel=plot_info[plot]["xlabel"],
        ylabel=plot_info[plot]["ylabel"],
        colocsceheme=plot_info[plot]['colorscheme'],
        # cols=3,
        save_dir=save_dir,
        save_name=file_name
    )


def plot_mptrj_energies(plot_info:dict, experiments:list, remove_outlier:bool, plot:str,
                     base_dir:Path, file_name:str, save_dir: Path|None,
                     ncols = None, add_fig_numbering:bool=True):

    mlip_key = plot_info[plot]['mlip_key']
    dft_key = plot_info[plot]['dft_key']

    print(f"{mlip_key = }, {dft_key = }")

    dft_energies, predicted_energies, rmses = [], [], []
    for exp in experiments:
        data_path = base_dir / exp / file_name    
        data = None
        with open(data_path, 'rb') as f:
            data_raw = pickle.load(f)
        plot_data = {dft_key:data_raw[dft_key] ,
                     mlip_key:data_raw[mlip_key], }
        data = pd.DataFrame(plot_data)
        min_dft = data[dft_key].min()
        max_dft = data[dft_key].max()
        rmse = get_rmse(data[dft_key], data[mlip_key])

        if remove_outlier:
            mask = data.eval(f'{mlip_key} > {min_dft} and {mlip_key} < {max_dft}')
            data = data[mask].copy()

        dft_energies.append(data[dft_key].values)
        predicted_energies.append(data[mlip_key].values)
        rmses.append(rmse)

    xmin = min(data[dft_key]) - 0.01
    xmax = max(data[dft_key])
    # xmax = 150

    fig = create_hist2d_subplots_with_single_cmap(
        dft_energy_list=dft_energies,
        predicted_energy_list=predicted_energies,
        model_names=experiments,
        xmin=xmin,
        xmax=xmax,
        rmses= rmses,
        xlabel=plot_info[plot]["xlabel"],
        ylabel=plot_info[plot]["ylabel"],
        colocsceheme=plot_info[plot]['colorscheme'],
        save_dir=save_dir,
        save_name="mptrj_mlip_dft_energy_subplots",
        cols=ncols,
        add_fig_numbering= add_fig_numbering
    )


def plot_mptrj_forces(plot_info:dict, experiments:list, remove_outlier:bool, plot:str,
                     base_dir:Path, file_name:str, save_dir: Path|None,
                      ncols=None ):
    print(f"{file_name = }")
    mlip_key = plot_info[plot]['mlip_key']
    dft_key = plot_info[plot]['dft_key']

    print(f"{mlip_key = }, {dft_key = }")

    dft_energies, predicted_energies, rmses = [], [], []
    for exp in experiments:
        data_path = base_dir / exp / file_name    
        data = None
        with open(data_path, 'rb') as f:
            data_raw = pickle.load(f)

        plot_data = {
                    dft_key:data_raw[dft_key] ,
                     mlip_key:data_raw[mlip_key]
                     }

        data = pd.DataFrame(plot_data)
        min_dft = data[dft_key].min()
        max_dft = data[dft_key].max()
        rmse = get_rmse(data[dft_key], data[mlip_key])

        if remove_outlier:
            mask = data.eval(f'{mlip_key} > {min_dft} and {mlip_key} < {max_dft}')
            data = data[mask].copy()

        dft_energies.append(data[dft_key].values)
        predicted_energies.append(data[mlip_key].values)
        rmses.append(rmse)

    xmin = min(data[dft_key]) - 0.01
    xmax = 100

    # fig = create_hist2d_subplots(
    fig = create_hist2d_subplots_with_single_cmap(
        dft_energy_list=dft_energies,
        predicted_energy_list=predicted_energies,
        model_names=experiments,
        xmin=xmin,
        xmax=xmax,
        rmses= rmses,
        xlabel=plot_info[plot]["xlabel"],
        ylabel=plot_info[plot]["ylabel"],
        colocsceheme=plot_info[plot]['colorscheme'],
        # cols=2,
        binsize=[100,100],
        save_dir=save_dir,
        save_name=f"mptrj_mlip_dft_energy_w_force_subplotsforces",
        cols=ncols
    )



###################################################################################################
###################################################################################################
# Properties of Fe
###################################################################################################
###################################################################################################

def create_bar_plot_seaborn_subplot(data: dict, xtick_labels: Sequence[str]=None, 
                                    ylabel:str='Values', 
                                    title:str='', ax=None, text_label=None):
    # If ax is not provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        standalone = False
    
    # Extract labels (keys) and values from the dictionary
    labels = list(data.keys())
    values = list(data.values())
    
    # Number of datasets and categories
    n_datasets = len(labels)
    n_categories = len(values[0])
    
    # Adjust width based on number of datasets (narrower for 5 datasets)
    width = 0.15 if n_datasets == 5 else 0.25
    
    # Set positions for bars
    x = range(n_categories)
    
    # Get a colorblind-friendly palette
    colors = sns.color_palette("colorblind", n_datasets)
    
    # Create bars for each dataset with colorblind-friendly colors
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.bar([p + i * width for p in x], val, width, label=label, color=colors[i])
    
    # Customize the plot
    ax.set_xlabel('Categories' if xtick_labels is None else None)
    if xtick_labels:
        ax.set_xticks([p + width * (n_datasets - 1) / 2 for p in x])
        ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title)
    if text_label is not None:
        ax.text(-0.1, 1.01, text_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='right')

    num_locations = len(labels) 
    # add hatches
    hatches = ['//', '---',  'oo','++','\\\\' '//']  
    count = 0
    for i, bar in enumerate(ax.patches):
        # print(f"{count = }, {i = }, {n_categories = }")
        bar.set_hatch(hatch=hatches[count])
        if (i+1) % n_categories == 0:
            count +=1

    # If standalone, save the plot and close
    if standalone:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(f'{save_plot_dir}_{title}.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()


def get_energy_volume_data_per_model(dat_model: pd.DataFrame):
    """
    Takes the dft values from 
    """

    # reference all plots to minimum to 0
    dat_model_min = dat_model['energy_per_atom'].min()
    dat_model['energy_per_atom'] -= dat_model_min

    # make it volume per atom
    dat_model['volume'] = dat_model['volume'] / 2

    return dat_model['energy_per_atom'].values, dat_model['volume'].values


def create_scatter_plot(data,x_vals, xlabel=None, ylabel='Values', title='Scatter Plot', ax=None, text_label=None):
    # If ax is not provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        standalone = False
    
    # Extract labels (keys) and values from the dictionary
    labels = list(data.keys())
    values = list(data.values())
    
    # Number of datasets
    n_datasets = len(labels)
    
    # Get a colorblind-friendly palette
    colors = sns.color_palette("colorblind", 5)
    markers = ["o", "o","o","o","o","o","o","o","o",]
    # Create scatter plot for each dataset
    for i, (label, val) in enumerate(zip(labels, values)):
        # Assuming values are paired (x, y) coordinates for scatter
        ax.scatter(x_vals, val, label=label, color=colors[i], alpha=0.6,
                   marker=markers[i])
    
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=14)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title)
    if text_label is not None:
        ax.text(-0.1, 1.01, text_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='right')
    
    # If standalone, save the plot and close
    if standalone:
        plt.tight_layout()
        if SAVE_PLOTS:
            print(f"Saving the plot in : {save_plot_dir}")
            plt.savefig(f'{save_plot_dir}/{title}.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()


# some functions to extract the properties
def get_b(c11,c12, c44):
    return (c11 + 2*c12)/3, (c11-c12)/2, c44

def get_cs(dat: pd.DataFrame):
    c11 = dat['C11'].values
    c12 = dat['C12'].values
    c44 = dat['C44'].values
    b1, b2, b3 = get_b(c11, c12, c44)
    return [b1[0], b2[0], b3[0]]

# for surfaces
def load_surface_data(file_path: Path):
    """
    Gives energies in J/m-2
    DFT energies are from [17] in sebastian paper
    """
    assert(isinstance(file_path, Path))

    dat = pd.read_csv(file_path)
    # print(dat.head(20))
    x_tick_label: Sequence[str] = ['(100)', '(110)', '(111)']
    bcc100 = dat['bcc100'].values[0]
    bcc110 = dat['bcc110'].values[0]
    bcc111 = dat['bcc111'].values[0]
    
    # the energies need to be converted from eV/A^2 to J/m^2
    bcc100 = 1.602176565e-19 * bcc100 / (1e-10)**2 
    bcc110 = 1.602176565e-19 * bcc110 / (1e-10)**2 
    bcc111 = 1.602176565e-19 * bcc111 / (1e-10)**2 

    return [bcc100, bcc110, bcc111]


def load_gb_data(file_path: Path, gb_type: str = 'tilt'):
    """
    THe data contains both twist and tilt grain boundaries
    the GB type to be plotted can be specified by the gb_type
    gb_type = 'tilt' or 'twist' by default it is tilt
    """
    assert(isinstance(file_path, Path))

    dat = pd.read_csv(file_path)
    dat = dat[dat['type']==gb_type]

    x_tick_label: Sequence[str] = dat['file_name'].values
    x_tick_label  = [i.strip('Fe-sigma') for i in x_tick_label]
    x_tick_label  = [i.strip('.cif') for i in x_tick_label]
    x_tick_label  = [i.replace('-',"",1) for i in x_tick_label]
    x_tick_label  = ['$\Sigma$'+i for i in x_tick_label]

    CHG2_energies =  dat['gb_energy'].values

    return x_tick_label, CHG2_energies


def convert_negative_to_bar(data_list):
    import re
    # Regular expression to match negative numbers (e.g., -1, -2, etc.) not followed by another digit
    pattern = r'-(\d)(\d*)'
    # Replace -num with $\bar{num}$
    return [re.sub(pattern, r'$\\bar{\1}\2$', item) for item in data_list]


def plot_fe_properties(experiments,base_dir, save_plot_dir=None, gb_type='Tilt' ):
    energy_volume_data = {}
    elastic_data = {}
    surface_data = {}
    gb_data = {}
    x_tick_labels_gb = None
    ev_xvalues = None

    for exp in experiments:

        ev_data_path = base_dir / exp / 'energy_volume.csv'
        data_ev = pd.read_csv(ev_data_path)
        energy_volume_data[exp], ev_xvalues = get_energy_volume_data_per_model(data_ev)

        elastic_data_path = base_dir / exp / "elastic_tensor.csv"
        data_elastic = pd.read_csv(elastic_data_path)
        elastic_data[exp] = get_cs(data_elastic)

        surface_data_path = base_dir / exp / "gb_surfaces.csv"
        surface_data[exp] = load_surface_data(surface_data_path)


        data_path = base_dir / exp / "gb_energies.csv"
        x_tick_labels_gb, gb_data[exp] = load_gb_data(data_path, gb_type=gb_type.lower())

    if gb_type =='Tilt':
       x_tick_labels_gb = convert_negative_to_bar(x_tick_labels_gb)

    fig, axes = plt.subplots(2,2 , figsize=(15, 10))

    for exp in experiments:
        print("----------------------------------------------------------------")
        print(f"{exp = }")

        if exp != 'DFT':
            error_elastic = np.abs(np.array(elastic_data[exp]) - np.array(elastic_data['DFT']))
            relatitve_error = np.divide(error_elastic, np.array(elastic_data['DFT']))
            print(f"Elastic properties error(b, c', c44): \n{relatitve_error*100 = } in percentage")

            error_gb =np.abs(np.mean(gb_data[exp] - gb_data['DFT']))
            print(f"GB_error (MAE) : {error_gb = }")
            error_sur =abs(np.mean(np.array(surface_data[exp]) - np.array(surface_data['DFT'])))
            print(f"surface_error (MAE): {error_sur = }")

    # energy volume curve
    create_scatter_plot(
        data=energy_volume_data,
        x_vals=ev_xvalues,
        xlabel='Volume ($\AA^3$/atom)',
        ylabel='Energy (eV/atom)',
        title='Energy Volume Curve', ax=axes[0, 0], text_label='a)')

    create_bar_plot_seaborn_subplot(
        data=elastic_data,
        xtick_labels=['B', 'C\'', '$C_{44}$'],
        ylabel='Elastic properties (GPa)',
        title='Elastic Properties',ax=axes[0, 1], text_label='b)')

    create_bar_plot_seaborn_subplot(
        data=surface_data,
        xtick_labels=['100', '110', '111'],
        ylabel='Surface Energy ($J/m^{-2}$)',
        title='Surface Energies',ax=axes[1, 1], text_label='d)')

    create_bar_plot_seaborn_subplot(
        data=gb_data,
        xtick_labels=x_tick_labels_gb,
        ylabel='GB Energy ($J/m^{-2}$)',
        title=f'{gb_type} Grain boundary Energies', ax=axes[1, 0], text_label='c)')

    handles, labels = axes[0,1].get_legend_handles_labels()

    # to adjust the space between the subplots
    plt.subplots_adjust( wspace=0.2, hspace=0.3)

    labels_new = []
    for label in labels:
        labels_new.append(label)

    # set the position of legends
    fig.legend(handles, labels_new, bbox_to_anchor=(0.5, 0.95), loc='upper center', borderaxespad=0., ncol=5, fontsize=12)

    for ax in fig.axes:
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
    if save_plot_dir:
        print("saving all properties subplot in ", save_plot_dir)
        plt.savefig(f"{save_plot_dir}/fe_properties_subplots.png", dpi=500,transparent=True)
    else:
        print("not saving all properties subplot")


def plot_gb_alone(experiments, base_dir, gb_type):
    """ It actually plots twist GB and saves it"""
    gb_data_tilt = {}
    x_tick_labels_gb_tilt = None
    for exp in experiments:
        data_path = base_dir / exp / "gb_energies.csv"
        x_tick_labels_gb_tilt, gb_data_tilt[exp] = load_gb_data(data_path, gb_type=gb_type)

    x_tick_labels_gb_tilt = convert_negative_to_bar(x_tick_labels_gb_tilt)
    print(x_tick_labels_gb_tilt)


    fig = plt.figure()
    ax_tilt = fig.add_subplot(111)
    create_bar_plot_seaborn_subplot(
        data=gb_data_tilt,
        title=f"{gb_type}",
        xtick_labels=x_tick_labels_gb_tilt,
        ylabel='GB Energy ($J/m^{-2}$))',
         ax=ax_tilt)
    

    plt.subplots_adjust(top=0.85)
    fig.legend(handles, labels_new, bbox_to_anchor=(0.5, 0.95), loc='upper center', borderaxespad=0., ncol=5, fontsize=12)
    # fig.tight_layout()
    fig.savefig(f'{save_plot_dir}/{gb_type}_GB.png', dpi=500)
    plt.show()
    plt.close(fig)



###################################################################################################
###################################################################################################
# impuritie interactions
###################################################################################################
###################################################################################################


def remove_elements_from_df(df, string_values, to_include:bool):
    """
    Remove rows from a DataFrame where specified string values appear in 'x' or 'y' columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns.
    string_values (list): List of string values to remove.
    
    Returns:
    pd.DataFrame: Filtered DataFrame with rows containing string_values removed.
    """
    # Ensure 'x' and 'y' columns exist
    if not {'X', 'Y'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    # Create a mask for rows where 'x' or 'y' contain any of the string values
    if to_include:
        mask = ((df['X'].isin(string_values)) | (df['Y'].isin(string_values)))
    else:
        mask = ~((df['X'].isin(string_values)) | (df['Y'].isin(string_values)))
    
    # Return filtered DataFrame
    return df[mask].copy()


def plot_impurities(experiments: list, save_dir: Path | None = None, base_dir: Path = Path('output'),
                    file_name: str = 'binding_energy.csv', elements_to_remove: list | None = None,
                    to_include: list | None = None, img_col:int = 6 ):

    if isinstance(experiments, Path):
        experiments = ['DFT', experiments.name]

    sns.set_context("notebook", font_scale=1.5)  # Increase font size globally (1.5x default)
    if elements_to_remove is not None:
        print(f"removing elements {elements_to_remove}")

    orig = None
    
    for exp in experiments:
        dat = pd.read_csv(base_dir / exp / file_name)
        if elements_to_remove is not None:
            xy = dat['x-y'].values
            dat['X'] = [i.split('-')[0] for i in xy]
            dat['Y'] = [i.split('-')[1] for i in xy]
            dat = remove_elements_from_df(dat, elements_to_remove, to_include=False)
        if to_include is not None:
            dat = remove_elements_from_df(dat, to_include, to_include=True)

        dat['model'] = [exp] * len(dat)

        if orig is None:
            orig = dat
        else:
            orig = pd.concat([orig, dat], ignore_index=True)

    # Check if orig is None or empty
    if orig is None or orig.empty:
        raise ValueError("No data to plot. Check experiments and file paths.")

    # Ensure the number of colors and markers matches the number of unique models
    unique_models = orig['model'].nunique()
    palette = sns.color_palette("colorblind", n_colors=unique_models)
    markers = ["X", "$\circ$","$\diamond$",  "o", "d",">", "^", "<", "X", "X", "X", "X", "X"]
    use_markers = markers[:min(unique_models, len(markers))]

    # Create a marker dictionary for consistent marker styles
    marker_dict = {model: marker for model, marker in zip(orig['model'].unique(), use_markers)}

    # Create FacetGrid
    g = sns.FacetGrid(orig, col="x-y", col_wrap=img_col, height=4, aspect=1.0)

    # Map scatterplot with larger marker size, disabling automatic legend
    g.map_dataframe(sns.scatterplot, x="nn", y="binding energy", hue="model",
                    style="model", markers=marker_dict, palette=palette, legend=False, s=200)

    # Set axis labels, titles, and ticks with larger font sizes
    g.set(xticks=[1, 2, 3, 4, 5])
    g.set_axis_labels("nn", "binding energy (eV)", fontsize=20)
    g.set_titles(col_template="{col_name}", size=18)
    for ax in g.axes.flat:
        ax.tick_params(axis='both', labelsize=14)  # Tick labels
        ax.axhline(y=0, xmin=0, xmax=5, color='black', linestyle='--', lw=0.5)

    # Create a single legend with unique model values in a single row at the top
    handles, labels = [], []
    for model in orig['model'].unique():
        color = palette[list(orig['model'].unique()).index(model)]
        marker = marker_dict[model]
        handles.append(plt.Line2D([0], [0], marker=marker, color='w', label=model,
                                  markerfacecolor=color, 
                                  markersize=12))
        labels.append(model)
    
    ## below snippet added for borders
    for ax in g.axes.flatten():
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
    # Add the legend at the top, in a single row
    g.figure.legend(handles=handles, labels=labels, title="Model", title_fontsize=22,
                    fontsize=20, ncol=unique_models,  # Single row with all models
                    bbox_to_anchor=(0.5, 0.90), loc='lower center', borderaxespad=0.)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # when using the whole plot
    g.figure.subplots_adjust(top=0.89)  # Reserve space at the top for the legend

    # Save the plot if save_dir is provided
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = ""
        if to_include:
            for st in to_include:
                suffix += st
        print(f"saving figures to :{save_dir}")
        plt.savefig(save_dir / f'imp_imp_interactions_{suffix}.png', bbox_inches='tight',transparent=True )
        plt.savefig(save_dir / f'imp_imp_interactions_{suffix}.svg', bbox_inches='tight',transparent=True )
    
    plt.show()


def get_1nn_MAE(experiments, elements_to_remove,
                base_dir, to_include=None,
                file_name = "binding_energy.csv"):
    """
    Despite the name it prints RMSE for all nn exluding the
    elements_to_remove and including only the to_include elements
    # NOTE:
    Assumes first element in experiments is DFT. at gets the RMSE from 
    the this.
    """
    orig = None
    for exp in experiments:
        dat = pd.read_csv(base_dir / exp / file_name)
        if elements_to_remove is not None:
            xy = dat['x-y'].values
            dat['X'] = [i.split('-')[0] for i in xy]
            dat['Y'] = [i.split('-')[1] for i in xy]
            dat = remove_elements_from_df(dat, elements_to_remove, to_include=False)
        if to_include is not None:
            dat = remove_elements_from_df(dat, to_include, to_include=True)

        if orig is None:
            orig = dat
        else:
            orig[exp] = dat['binding energy'].values

    for exp in experiments:
        if exp !='DFT':
            error = abs(orig['binding energy'].values - orig[exp].values)
            mae = sum(error)/len(error)
            rmse = get_rmse(orig['binding energy'].values, orig[exp].values)
            print(f"{exp = }:\n \t {mae = }\t {rmse = }")

