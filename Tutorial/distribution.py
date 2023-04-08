import numpy as np

class ReconstructedDistribution:
    def __init__(self, max_tau):
        self.max_tau = max_tau
        self.set_initial_state()

    def set_initial_state(self):
        self.tau = np.random.uniform(0,self.max_tau)
        self.tau_occurrences = [self.tau]

    def eval_detailed_balance(self, new_tau):
        prob_new_tau = np.exp(-new_tau)
        prob_curr_tau = np.exp(-self.tau)
        return prob_new_tau/prob_curr_tau

    def metropolis(self,ratio_acceptances_prob):
        return min(1, ratio_acceptances_prob)

    def eval_change_tau(self):
        new_tau = np.random.uniform(0,self.max_tau)
        ratio_acceptances_prob = self.eval_detailed_balance(new_tau)
        acceptance_new = self.metropolis(ratio_acceptances_prob)
        if acceptance_new == 1:
            self.change_tau(new_tau)
        elif 0 <= acceptance_new < 1:
            random_sample = np.random.uniform(0,1)
            if random_sample <= acceptance_new:
                self.change_tau(new_tau)
                
    def change_tau(self, new_tau):
        self.tau = new_tau

    def update_tau_occurrences(self):
        self.tau_occurrences.append(self.tau)
