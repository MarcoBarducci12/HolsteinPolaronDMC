import argparse
import random
from dmc import runDiagrammaticMonteCarlo
import plot
from polaron import Polaron

"""From command line we can pass general values for the parameters of the
   simulation such as the energy of both electron and phonon and the
   intensity of the electron phonon interaction. We can fix the "flight time"
   of the electron and the order of the initial diagram.
"""
parser = argparse.ArgumentParser(prog="""Diagrammatic MonteCarlo for Holstein polaron""",
                                 description="""Perform sampling of Feynman
                                diagrams for the Holstein polaron in a
                                tight binding approach""")
parser.add_argument('--nsteps', dest='nsteps', type=int, default=10000,
                help="Number of MonteCarlo samples")
parser.add_argument('--order', dest='order', type=int, default=0,
                help="order of the initial diagram")
parser.add_argument('--mu', dest='mu', type=float, default=0.0,
                help="energy of the electron")
parser.add_argument('--omega', dest='omega', type=float, default=1.0,
                help="phonon energy")
parser.add_argument('--g', dest='g', type=float, default=0.3,
                help="electron phonon coupling constant")
parser.add_argument('--time', dest='time_scaling', type=float, default=1.0,
                help="time of electron propagator")
parser.add_argument('--max_time', dest='max_time', type=int, default=30,
                help="upper bound for lifetime of electron propagator")

if __name__ == "__main__":

    args=parser.parse_args()
    polaron = Polaron(args)
    diagrams_info = runDiagrammaticMonteCarlo(polaron, args)
    plot.plotMonteCarlo(diagrams_info)
