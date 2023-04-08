"""Module that contains utilities functions to display plots and useful
quantities"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def eval_mean_energy(energy_sequence : list) -> float :
    """Parameters: energy_sequence, list of energy evaluated for each
    diagram samped
    Return: mean energy value evaluated over all diagrams sampled
    """
    return np.mean(np.array(energy_sequence))

def eval_mean_order(order_sequence : list) -> float :
    """Parameters: order_sequence, list of order of each diagram sampled
    Return: mean order evaluated over all diagrams sampled
    """
    return np.mean(np.array(order_sequence))

def get_bins_edges(order_sequence : list) -> list:
    """Return a list of left bin edges and right edge of last bin.
    Each bin is centered on each of the possible values"""
    n_bins = max(order_sequence) + 1
    bins_edges = np.arange(n_bins+1) - 0.5
    return bins_edges

def plot_montecarlo(diagrams_info : dict):
    """ Allow use of LaTeX font in graphics
        Return an histogram:
       -optimized binning divisions
       -occurrences for each bin are normalized to the number of samples
       -plot the sampled probability distribution vs the analitycal one
    """
    plt.rcParams['text.usetex'] = True
    n , bins, patches = plt.hist(x=diagrams_info['Order_sequence'], density=True,
                        bins=get_bins_edges(diagrams_info['Order_sequence']),
                        ec='black', fc='blue', alpha=0.8)

    """Define labels entry for legend"""
    mean_order = eval_mean_order(diagrams_info['Order_sequence'])
    order_label = 'Mean diagram order: ' + f'{mean_order:.5f}'

    mean_energy = eval_mean_energy(diagrams_info['Energy_sequence'])
    energy_label = 'Mean energy: ' + f'{mean_energy:.5f}'

    invalid_diagrams = diagrams_info['Invalid_diagrams']
    inv_diag_label = 'Number of invalid diagrams: ' + f'{invalid_diagrams}'
    legend_elements = [Line2D([0], [0], color='b', label=order_label),
                       Line2D([0], [0], color='r', label=energy_label),
                       Line2D([0], [0], color='g', label=inv_diag_label)]

    plt.xlabel('Diagram order')
    plt.ylabel(r'Sampled probability distribution')
    plt.yscale("log")
    plt.legend(handles=legend_elements)
    plt.savefig("DiagramOrderDistribution.png")
    plt.show()

def plot_green_function(tau_sequence : list):
    """ Allow use of LaTeX font in graphics
        Return an histogram:
       -optimized binning divisions
       -occurrences for each bin are normalized to the number of samples
       -plot the sampled probability distribution vs the analitycal one
    """
    plt.rcParams['text.usetex'] = True
    plt.hist(x=tau_sequence, density=True, bins=50,
             ec='black', fc='blue', alpha=0.8, label=r"Sampled $G(\tau)$")

    plt.xlabel(r'Lifetime $(\tau)$')
    plt.ylabel(r'$G(\tau)$')
    plt.savefig("GreenFunctionHistogram.png")
    plt.show()
