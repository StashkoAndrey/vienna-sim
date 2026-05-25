# phage_5.py
#
# Same logic as original phage_5.py, but imports from parameters_extracted.py
# so bacteria, grid, main, plots, and phages use the same active parameter file.

import random
import numpy as np
from parameters_extracted_5 import (
    PHAGE_A_ADSORPTION_RATE, PHAGE_B_ADSORPTION_RATE, PHAGE_C_ADSORPTION_RATE,
    PHAGE_A_BURST_SIZE, PHAGE_B_BURST_SIZE, PHAGE_C_BURST_SIZE,
    PHAGE_A_DECAY_RATE, PHAGE_B_DECAY_RATE, PHAGE_C_DECAY_RATE,
    PHAGE_A_LATENT_PERIOD, PHAGE_B_LATENT_PERIOD, PHAGE_C_LATENT_PERIOD,
    INITIAL_PHAGE_A_CONCENTRATION,
    INITIAL_PHAGE_B_CONCENTRATION,
    INITIAL_PHAGE_C_CONCENTRATION,
)

ADSORPTION = {'A': PHAGE_A_ADSORPTION_RATE, 'B': PHAGE_B_ADSORPTION_RATE, 'C': PHAGE_C_ADSORPTION_RATE}
BURST_SIZE = {'A': PHAGE_A_BURST_SIZE, 'B': PHAGE_B_BURST_SIZE, 'C': PHAGE_C_BURST_SIZE}
DECAY_RATE = {'A': PHAGE_A_DECAY_RATE, 'B': PHAGE_B_DECAY_RATE, 'C': PHAGE_C_DECAY_RATE}
LATENT = {'A': PHAGE_A_LATENT_PERIOD, 'B': PHAGE_B_LATENT_PERIOD, 'C': PHAGE_C_LATENT_PERIOD}


def make_empty_phage_fields(size):
    phA = np.full((size, size), INITIAL_PHAGE_A_CONCENTRATION, dtype=np.float32)
    phB = np.full((size, size), INITIAL_PHAGE_B_CONCENTRATION, dtype=np.float32)
    phC = np.full((size, size), INITIAL_PHAGE_C_CONCENTRATION, dtype=np.float32)
    return phA, phB, phC


def _infection_attempt(local_phage, channels, adsorption_rate):
    if local_phage <= 0 or channels == 0:
        return False

    p0 = adsorption_rate * local_phage
    p0 = max(0.0, min(1.0, p0))

    p_infect = 1.0 - (1.0 - p0) ** channels
    p_infect = max(0.0, min(1.0, p_infect))

    return random.random() < p_infect


def attempt_new_infections(grid, pending_lysis):
    size = grid.size

    for i in range(size):
        for j in range(size):
            cell = grid.grid[i][j]
            if cell is None:
                continue

            local = {
                'A': float(grid.phages_A[i, j]),
                'B': float(grid.phages_B[i, j]),
                'C': float(grid.phages_C[i, j]),
            }
            channels = {
                'A': cell.channels_A,
                'B': cell.channels_B,
                'C': cell.channels_C,
            }

            successes = [
                phage_type
                for phage_type in ('A', 'B', 'C')
                if _infection_attempt(local[phage_type], channels[phage_type], ADSORPTION[phage_type])
            ]

            if successes:
                phage_type = random.choice(successes)
                pending_lysis.append({
                    'i': i,
                    'j': j,
                    'phage_type': phage_type,
                    'steps_left': LATENT[phage_type],
                })


def process_pending_lysis(grid, pending_lysis):
    finished = []

    for idx, rec in enumerate(pending_lysis):
        rec['steps_left'] -= 1

        if rec['steps_left'] <= 0:
            i = rec['i']
            j = rec['j']
            phage_type = rec['phage_type']

            if grid.grid[i][j] is not None:
                grid.grid[i][j] = None
                getattr(grid, f'phages_{phage_type}')[i, j] += BURST_SIZE[phage_type]

            finished.append(idx)

    for idx in reversed(finished):
        pending_lysis.pop(idx)


def decay_phages(grid):
    grid.phages_A *= (1.0 - DECAY_RATE['A'])
    grid.phages_B *= (1.0 - DECAY_RATE['B'])
    grid.phages_C *= (1.0 - DECAY_RATE['C'])
