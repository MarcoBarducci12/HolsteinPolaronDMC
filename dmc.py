"""Helper function that perform the diagrammatic MonteCarlo simulation for the
Holstein polaron. """

import argparse
import random
from polaron import Polaron


def run_diagrammatic_montecarlo(polaron : Polaron, args : argparse.Namespace) -> dict :
    """Input parameter:
    - args : list that contains the fundamental parameters for the simulation
    """
    for _ in range(1, args.nsteps):
        try:
            if polaron.diagram['order'] == 0:
                number = random.randrange(len(polaron.zero_order_updates))
                polaron.zero_order_updates[number]()
            else:
                number = random.randrange(len(polaron.updates))
                polaron.updates[number]()
        except Exception as error:
            print(f"Invalid step in DMC, {error}, {error.__class__}")
            polaron.diagrams_info['Invalid_diagrams'] += 1
            continue
        polaron.eval_diagram_energy()
        polaron.diagrams_info['Order_sequence'].append(polaron.diagram['order'])
        polaron.diagrams_info['Energy_sequence'].append(polaron.diagram['total_energy'])
        polaron.diagrams_info['Tau_sequence'].append(polaron.diagram['time_scaling'])
    return polaron.diagrams_info
