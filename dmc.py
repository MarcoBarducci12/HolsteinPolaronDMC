import random
import time
import numpy as np
import argparse

def create_initial_diagram(args : argparse.Namespace):
    """Produce initial diagram for Holstein polaron in the tight
    binding limit.
    Default value are provided if the user does not write
    command line values for the simulation parameters
    """
    diagram = {'order': args.order,
               'phonon_energy': args.omega,
               'phonon_list': [],
               'electron_energy': args.mu,
               'electron_gen_time': 0,
               'electron_rem_time': 1,
               'ep_coupling': args.g,
               'time_scaling': args.time_scaling,
               'total_energy': 0}
    return diagram


def generate_phonon() -> dict:
    """Produce phonon propagator extracting scaled generation
    and removal time from uniform distribution.
    """
    t_gen = np.random.uniform(0, 1)
    t_rem = np.random.uniform(t_gen, 1)
    phonon = {'gen_time': t_gen, 'rem_time': t_rem}
    return phonon

def get_phonon(diagram: dict) -> tuple :
    """Retrieve randomly a phonon from the one in the diagram"""
    phonon_tag = random.randrange(len(diagram['phonon_list']))
    return (diagram['phonon_list'][phonon_tag], phonon_tag)

def add_phonon_scaling(diagram):
    """Evaluate scaling parameter due to electron phonon coupling"""
    return (diagram['ep_coupling']*diagram['time_scaling'])**2


def weigth_ratio_add(diagram, phonon):
    """Evaluate weigth_ratio between proposed and current Feynman diagram"""
    phonon_propagator = np.exp(-diagram['time_scaling']*diagram['phonon_energy'] *
                           (phonon['rem_time'] - phonon['gen_time']))
    if abs(add_phonon_scaling(diagram)*phonon_propagator) == 0.0:
        raise ValueError("Underflow error in evaluating phonon propagator\n")

    return add_phonon_scaling(diagram)*phonon_propagator


def weigth_ratio_remove(diagram, phonon):
    """Evaluate weigth_ratio between proposed and current Feynman diagram"""
    phonon_propagator = np.exp(diagram['time_scaling']*diagram['phonon_energy']*
                            (phonon['rem_time'] - phonon['gen_time']))
    if np.isinf(phonon_propagator/add_phonon_scaling(diagram)) == True:
        raise ValueError("Overflow Error in evaluating phonon propagator\n")

    return phonon_propagator/add_phonon_scaling(diagram)



def proposal_add_ratio(diagram, phonon):
    """Evaluate ratio between p_reverse and p_current
    i.e. removing the phonon added and adding it for the
    current update.
    p_current: uniform gen_time between 0 and 1 *
    uniform rem_time between gen_time and 1
    p_reverse: removal of the phonon whose probability is 1/(# of phonons)
    """
    return (1-phonon['gen_time'])/(diagram['order']+1)


def proposal_remove_ratio(diagram, phonon):
    """Evaluate ratio between p_reverse and p_current
    i.e. adding the considered phonon and removing it
    p_current: uniform gen_time between 0 and 1 *
    uniform rem_time between gen_time and 1
    p_reverse: removal of the phonon whose probability is 1/(# of phonons)
    """
    return diagram['order']/(1-phonon['gen_time'])


def add_internal(diagram, phonon):
    """Add a phonon to the diagram and update the order"""
    diagram['phonon_list'].append(phonon)
    diagram['order'] += 1


def remove_internal(diagram, phonon_tag):
    """Remove a phonon to the diagram and update the order"""
    del diagram['phonon_list'][phonon_tag]
    diagram['order'] -= 1


def metropolis(prob):
    """Using metropolis choice we ensure detailed balance for Markov chain"""
    acceptance = min(1, prob)
    return acceptance


def eval_add_internal(diagram):
    """Evaluate acceptance probability of internal phonon propagator and eventually
    add it to the diagram"""
    phonon = generate_phonon()
    try:
        prob = weigth_ratio_add(diagram, phonon) * \
            proposal_add_ratio(diagram, phonon)
    except ValueError as e:
        print(f"ValueError: {e}, {e.__class__}")
        raise ValueError(f"Add internal will not be performed in DMC")
    else:
        acceptance = metropolis(prob)
        if acceptance == 1:
            add_internal(diagram, phonon)
        elif 0 <= acceptance < 1:
            sample = np.random.uniform()
            if sample <= acceptance:
                add_internal(diagram, phonon)


def eval_remove_internal(diagram):
    """Choose one of the phonons randomly and evaluate
       the acceptance probability for the removal update
    """
    phonon, phonon_tag = get_phonon(diagram)
    try:
        acceptance_prob = weigth_ratio_remove(diagram, phonon) * \
            proposal_remove_ratio(diagram, phonon)
    except ValueError as e:
        print(f"ValueError: {e}, {e.__class__}")
        raise ValueError(f"Remove internal will not be performed in DMC")
    else:
        acceptance = metropolis(acceptance_prob)
        if acceptance == 1:
            remove_internal(diagram, phonon_tag)
        elif 0 <= acceptance < 1 :
            sample = np.random.uniform()
            if sample <= acceptance:
                remove_internal(diagram, phonon_tag)


def eval_diagram_energy(diagram):
    """Evaluate energy of the system at a certain iteration"""
    if diagram['order'] == 0:
        diagram['total_energy'] = 0
    else:
        energy = 0
        for phonon in diagram['phonon_list']:
            energy += diagram['phonon_energy']*(phonon['rem_time']-phonon['gen_time'])
        energy = (energy - diagram['order'])/diagram['time_scaling']
        diagram['total_energy'] = energy


def runDiagrammaticMonteCarlo(args) -> dict :
    """Input parameter:
       - args : list that contains the fundamental parameters for the simulation
       Output parameter:
       - diagrams_info : dictionary that contains the list of diagram order and
        diagram energy for each sample
    """
    updates = [eval_add_internal, eval_remove_internal]
    diagram = create_initial_diagram(args)
    diagrams_info = {'Order_sequence' : [diagram['order']],
                     'Energy_sequence': [diagram['total_energy']],
                     'Invalid_diagrams': 0}
    start = time.time()
    for _ in range(1, args.nsteps):
        try:
            if diagram['order'] == 0:
                eval_add_internal(diagram)
            else:
                number = random.randint(0, len(updates)-1)
                updates[number](diagram)
        except Exception as e:
            print(f"Invalid step in DMC")
            diagrams_info['Invalid_diagrams'] += 1
            continue

        eval_diagram_energy(diagram)
        diagrams_info['Order_sequence'].append(diagram['order'])
        diagrams_info['Energy_sequence'].append(diagram['total_energy'])
    end = time.time()
    print(end - start)
    return diagrams_info
