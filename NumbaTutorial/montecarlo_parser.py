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
        self.parser.add_argument('--nsteps', dest='nsteps', type=int, default=1000000,
                        help="Number of MonteCarlo steps (samples)")
        self.parser.add_argument('--max_time', dest='max_time', type=int, default=5,
                        help="Upper bound for the sampled values")
