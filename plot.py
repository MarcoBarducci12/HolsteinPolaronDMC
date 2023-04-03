"""Module that contains utilities functions to display plots and useful
quantities"""

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

def plotMonteCarlo(diagrams_info : dict):
    """ Allow use of LaTeX font in graphics
        Return an histogram:
       -optimized binning divisions
       -occurrences for each bin are normalized to the number of samples
       -plot the sampled probability distribution vs the analitycal one
    """
    plt.rcParams['text.usetex'] = True
    n , bins, patches = plt.hist(x=diagrams_info['Order_sequence'], density=True, bins='auto',
             color='blue', alpha=0.8)

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
