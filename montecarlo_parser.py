"""Helper class that store an argparse object initialized in the constructor
of a MonteCarloParser object"""

import argparse

class MonteCarloParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog="""Markov Chain MonteCarlo to
            reconstruct a distribution through an histogram""",
            description="""In this program we reconstruct an exponential
            probability distribution sampling the occurrences extracted
            from the target. We employ a Metropolis algorithm to sample
            different values from the distribution.""")
        self.parser.add_argument('--nsteps', dest='nsteps', type=int, default=10000,
                        help="Number of MonteCarlo steps (samples)")
        self.parser.add_argument('--nsteps_burn', dest='nsteps_burn', type=int, default=10000,
                        help="Number of thermalization steps for the Markov Chain")
        self.parser.add_argument('--order', dest='order', type=int, default=0,
                        help="Order of the initial diagram")
        self.parser.add_argument('--mu', dest='mu', type=float, default=0.0,
                        help="Energy of the electron")
        self.parser.add_argument('--omega', dest='omega', type=float, default=1.0,
                        help="Phonon energy")
        self.parser.add_argument('--g', dest='g', type=float, default=0.3,
                        help="Electron phonon coupling constant")
        self.parser.add_argument('--time', dest='time_scaling', type=float, default=1.0,
                        help="Initial lifetime of electron propagator")
        self.parser.add_argument('--max_time', dest='max_time', type=int, default=50,
                        help="Upper bound for lifetime of electron propagator")
