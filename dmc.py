"""Main function that perform the diagrammatic MonteCarlo simulation for the
Holstein polaron. """

import argparse
from polaron import Polaron
import random

def runDiagrammaticMonteCarlo(polaron : Polaron, args : argparse.Namespace) -> dict :
    """Input parameter:
    - args : list that contains the fundamental parameters for the simulation
    """
    for _ in range(1, args.nsteps):
        try:
            if polaron.diagram['order'] == 0:
                polaron.eval_add_internal()
            else:
                number = random.randrange(len(polaron.updates))
                polaron.updates[number]()
        except Exception as e:
            print(f"Invalid step in DMC, {e}, {e.__class__}")
            polaron.diagrams_info['Invalid_diagrams'] += 1
            continue
        polaron.eval_diagram_energy()
        polaron.diagrams_info['Order_sequence'].append(polaron.diagram['order'])
        polaron.diagrams_info['Energy_sequence'].append(polaron.diagram['total_energy'])
    #end = time.time()
    #print(end - start)
    return polaron.diagrams_info
