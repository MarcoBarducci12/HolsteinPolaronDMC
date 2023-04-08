"""Simple program that counts the occurrences sampled from a probability
distribution and reconstructs it through an histogram.
From command line we can pass the upper limit for the values that can be
extracted from the distribution we are reconstructing. We pass the number of steps
for the simulation.
"""
from distribution import ReconstructedDistribution
from montecarlo_parser import MonteCarloParser
from plot import plot_distribution


def run_montecarlo(distribution, nsteps):
    for _ in range(1, nsteps):
        distribution.eval_change_tau()
        distribution.update_tau_occurrences()
    return distribution.tau_occurrences

if __name__ == "__main__":

    mc_parser=MonteCarloParser()
    args=mc_parser.parser.parse_args()
    dist = ReconstructedDistribution(args.max_time)
    occurrences = run_montecarlo(dist, args.nsteps)
    plot_distribution(occurrences, args.max_time)
