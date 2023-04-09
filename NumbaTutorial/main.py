from distribution import Distribution
from montecarlo_parser import MonteCarloParser
from numba import njit
from plot import plot_distribution

@njit
def montecarlo(dist, steps):
    for _ in range(1,steps):
        dist.eval_change_tau()
        dist.update_occurrences()
    return dist

if __name__ == "__main__" :

    mc_parser=MonteCarloParser()
    args=mc_parser.parser.parse_args()
    dist = Distribution(args.max_time)
    reconstructed_dist = montecarlo(dist,args.nsteps)
    plot_distribution(reconstructed_dist.tau_occurrences,args.max_time)
