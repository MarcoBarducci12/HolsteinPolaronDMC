"""Plot function to display the reconstructed exponetially decaying probability
distribution"""
import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(occurrences : list, max_time):
    """ Allow use of LaTeX font in graphics
        Return an histogram:
       -optimized binning divisions
       -occurrences for each bin are normalized to the number of samples
       -plot the sampled probability distribution vs the analitycal one
    """
    plt.rcParams['text.usetex'] = True
    plt.hist(x=occurrences, density=True, bins=50, 
             ec='black', fc='blue', alpha=0.8, label="Sampled distribution")

    x_val=np.linspace(0, max_time, 100)
    plt.plot(x_val, np.exp(-x_val), color='red', label="Target distribution")

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$Q(\tau)$')
    plt.legend()
    plt.savefig("ProbabilityHistogram.png")
    plt.show()
