import random
import numpy as np
import argparse

class Polaron:
    def __init__(self, args: argparse.Namespace):
        self.create_initial_diagram(args)
        self.set_updates()
        self.set_diagrams_info()

    def create_initial_diagram(self, args : argparse.Namespace):
        """Produce initial diagram for Holstein polaron in the tight
        binding limit.
        Default value are provided if the user does not write
        command line values for the simulation parameters
        """
        self.diagram = {'order': args.order,
                   'phonon_energy': args.omega,
                   'phonon_list': [],
                   'electron_energy': args.mu,
                   'electron_gen_time': 0,
                   'electron_rem_time': 1,
                   'ep_coupling': args.g,
                   'time_scaling': args.time_scaling,
                   'total_energy': 0}

    def set_updates(self):
        """Fill the list of possible updates to the diagram"""
        self.updates = [self.eval_add_internal, self.eval_remove_internal]

    def set_diagrams_info(self):
        """Set the initial values for diagram's info like initial order,
        energy and number of invalid diagrams"""
        self.diagrams_info = {'Order_sequence' : [self.diagram['order']],
                             'Energy_sequence': [self.diagram['total_energy']],
                             'Invalid_diagrams': 0}

    def metropolis(self, prob):
        """Using metropolis choice we ensure detailed balance for Markov chain"""
        acceptance = min(1, prob)
        return acceptance

    def add_phonon_scaling(self):
        """Evaluate scaling parameter due to electron phonon coupling"""
        return (self.diagram['ep_coupling']*self.diagram['time_scaling'])**2

    def weigth_ratio_add(self, phonon):
        """Evaluate weigth_ratio between proposed and current Feynman diagram"""
        phonon_propagator = np.exp(-self.diagram['time_scaling']*self.diagram['phonon_energy'] *
                            (phonon['rem_time'] - phonon['gen_time']))
        if abs(self.add_phonon_scaling()*phonon_propagator) == 0.0:
            raise ValueError("Underflow error in evaluating phonon propagator\n")

        return self.add_phonon_scaling()*phonon_propagator

    def proposal_add_ratio(self, phonon):
        """Evaluate ratio between p_reverse and p_current
        i.e. removing the phonon added and adding it for the
        current update.
        p_current: uniform gen_time between 0 and 1 *
        uniform rem_time between gen_time and 1
        p_reverse: removal of the phonon whose probability is 1/(# of phonons)
        """
        return (1-phonon['gen_time'])/(self.diagram['order']+1)

    def add_internal(self, phonon):
        """Add a phonon to the diagram and update the order"""
        self.diagram['phonon_list'].append(phonon)
        self.diagram['order'] += 1

    def eval_add_internal(self):
        """Evaluate acceptance probability of internal phonon propagator and eventually
        add it to the diagram"""
        phonon = self.generate_phonon()
        try:
            prob = self.weigth_ratio_add(phonon) * \
                self.proposal_add_ratio(phonon)
        except ValueError as error:
            print(f"ValueError: {error}, {error.__class__}")
            raise ValueError("Add internal will not be performed in DMC") from error

        acceptance = self.metropolis(prob)
        if acceptance == 1:
            self.add_internal(phonon)
        elif 0 <= acceptance < 1:
            sample = np.random.uniform()
            if sample <= acceptance:
                self.add_internal(phonon)

    def generate_phonon(self) -> dict:
        """Produce phonon propagator extracting scaled generation
        and removal time from uniform distribution.
        """
        t_gen = np.random.uniform(0, 1)
        t_rem = np.random.uniform(t_gen, 1)
        return {'gen_time': t_gen, 'rem_time': t_rem}

    def get_phonon(self) -> tuple :
        """Retrieve randomly a phonon from the one in the diagram"""
        phonon_tag = random.randrange(len(self.diagram['phonon_list']))
        return (self.diagram['phonon_list'][phonon_tag], phonon_tag)

    def weigth_ratio_remove(self, phonon):
        """Evaluate weigth_ratio between proposed and current Feynman diagram"""
        phonon_propagator = np.exp(self.diagram['time_scaling']*self.diagram['phonon_energy']*
                                (phonon['rem_time'] - phonon['gen_time']))
        if np.isinf(phonon_propagator/self.add_phonon_scaling()) is True:
            raise ValueError("Overflow Error in evaluating phonon propagator\n")

        return phonon_propagator/self.add_phonon_scaling()

    def proposal_remove_ratio(self, phonon):
        """Evaluate ratio between p_reverse and p_current
        i.e. adding the considered phonon and removing it
        p_current: uniform gen_time between 0 and 1 *
        uniform rem_time between gen_time and 1
        p_reverse: removal of the phonon whose probability is 1/(# of phonons)
        """
        return self.diagram['order']/(1-phonon['gen_time'])

    def remove_internal(self, phonon_tag):
        """Remove a phonon to the diagram and update the order"""
        del self.diagram['phonon_list'][phonon_tag]
        self.diagram['order'] -= 1

    def eval_remove_internal(self):
        """Choose one of the phonons randomly and evaluate
        the acceptance probability for the removal update
        """
        phonon, phonon_tag = self.get_phonon()
        try:
            acceptance_prob = self.weigth_ratio_remove(phonon) * \
                self.proposal_remove_ratio(phonon)
        except ValueError as error:
            print(f"ValueError: {error}, {error.__class__}")
            raise ValueError("Remove internal will not be performed in DMC") from error

        acceptance = self.metropolis(acceptance_prob)
        if acceptance == 1:
            self.remove_internal(phonon_tag)
        elif 0 <= acceptance < 1 :
            sample = np.random.uniform()
            if sample <= acceptance:
                self.remove_internal(phonon_tag)

    def eval_diagram_energy(self):
        """Evaluate energy of the system at a certain iteration"""
        if self.diagram['order'] == 0:
            self.diagram['total_energy'] = 0
        else:
            energy = 0
            for phonon in self.diagram['phonon_list']:
                energy += self.diagram['phonon_energy']*(phonon['rem_time']-phonon['gen_time'])
            energy = (energy - self.diagram['order'])/self.diagram['time_scaling']
            self.diagram['total_energy'] = energy
