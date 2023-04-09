import numpy as np
from numba.experimental import jitclass
from numba import types, typed

spec = [('tau', types.float32),
        ('max_tau', types.uint8),
        ('tau_occurrences', types.ListType(types.float64))]

@jitclass(spec)
class Distribution:

    def __init__(self, max_tau):
        self.max_tau = max_tau
        self.tau = np.random.uniform(0, self.max_tau)
        self.tau_occurrences=typed.List.empty_list(types.float64)
        self.tau_occurrences.append(self.tau)

    def detailed_balance(self, new_tau):
        prob_new = np.exp(-new_tau)
        prob_curr = np.exp(-self.tau)
        return prob_new/prob_curr

    def metropolis(self, ratio_acceptance):
        return min(1, ratio_acceptance)

    def eval_change_tau(self):
        new_tau = np.random.uniform(0, self.max_tau)
        ratio_acceptance = self.detailed_balance(new_tau)
        acceptance = self.metropolis(ratio_acceptance)
        if acceptance == 1:
            self.tau = new_tau
        elif 0 <= acceptance < 1:
            sample = np.random.uniform(0,1)
            if acceptance >= sample:
                self.tau = new_tau

    def update_occurrences(self):
        self.tau_occurrences.append(self.tau)
