"""From command line we can pass general values for the parameters of the
   simulation such as the energy of both electron and phonon and the
   intensity of the electron phonon interaction. We can fix the "flight time"
   of the electron and the order of the initial diagram.
"""

import argparse
import random
import plot
from montecarlo_parser import MonteCarloParser
from dmc import run_diagrammatic_montecarlo
from polaron import Polaron


if __name__ == "__main__":

    mc_parser=MonteCarloParser()
    args=mc_parser.parser.parse_args()
    polaron = Polaron(args)
    diagrams_info = run_diagrammatic_montecarlo(polaron, args)
    plot.plot_montecarlo(diagrams_info)
    plot.plot_green_function(diagrams_info['Tau_sequence'])
