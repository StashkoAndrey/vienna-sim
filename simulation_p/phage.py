# phage.py

import numpy as np
import random
from parameters_p import (
    PHAGE_A_ADSORPTION_RATE,
    PHAGE_A_BURST_SIZE,
    PHAGE_A_LATENT_PERIOD,
    PHAGE_A_DECAY_RATE,
    PHAGE_B_ADSORPTION_RATE,
    PHAGE_B_BURST_SIZE,
    PHAGE_B_LATENT_PERIOD,
    PHAGE_B_DECAY_RATE,
    INITIAL_PHAGE_A_CONCENTRATION,
    INITIAL_PHAGE_B_CONCENTRATION,
)

def make_empty_phage_fields(size):
    """
    Returns two NumPy arrays of shape (size, size):
     - phages_A (initialized to INITIAL_PHAGE_A_CONCENTRATION everywhere)
     - phages_B (initialized to INITIAL_PHAGE_B_CONCENTRATION everywhere)
    """
    phA = np.full((size, size), INITIAL_PHAGE_A_CONCENTRATION, dtype=np.float32)
    phB = np.full((size, size), INITIAL_PHAGE_B_CONCENTRATION, dtype=np.float32)
    return phA, phB

def attempt_new_infections(grid, pending_lysis):
    """
    For each occupied cell in grid.grid, compute two independent
    adsorption attempts—one for phage_A vs. channels_A, one for phage_B vs. channels_B—
    regardless of any 'type' field on the bacterium. If both phage types “win” in this
    timestep, choose one at random to actually infect.

    Each adsorption probability is computed as:
        p0_A = PHAGE_A_ADSORPTION_RATE * local_phA
        nA = bacterium.channels_A
        p_infectA = 1 - (1 - clamp(p0_A,0,1))^nA

        p0_B = PHAGE_B_ADSORPTION_RATE * local_phB
        nB = bacterium.channels_B
        p_infectB = 1 - (1 - clamp(p0_B,0,1))^nB

    If either succeeds, schedule that infection (with the corresponding latent period).
    If both succeed, pick A or B with equal chance.
    """
    size = grid.size

    for i in range(size):
        for j in range(size):
            bacterium = grid.grid[i][j]
            if bacterium is None:
                continue

            # --- look up local phage concentrations ---
            local_phA = float(grid.phages_A[i, j])
            local_phB = float(grid.phages_B[i, j])

            # --- compute infection probability for phage A ---
            infected_by_A = False
            nA = bacterium.channels_A
            if local_phA > 0 and nA > 0:
                p0_A = PHAGE_A_ADSORPTION_RATE * local_phA
                p0_A = min(max(p0_A, 0.0), 1.0)
                p_infectA = 1.0 - (1.0 - p0_A) ** nA
                p_infectA = min(p_infectA, 1.0)
                if random.random() < p_infectA:
                    infected_by_A = True

            # --- compute infection probability for phage B ---
            infected_by_B = False
            nB = bacterium.channels_B
            if local_phB > 0 and nB > 0:
                p0_B = PHAGE_B_ADSORPTION_RATE * local_phB
                p0_B = min(max(p0_B, 0.0), 1.0)
                p_infectB = 1.0 - (1.0 - p0_B) ** nB
                p_infectB = min(p_infectB, 1.0)
                if random.random() < p_infectB:
                    infected_by_B = True

            # --- resolve simultaneous successes ---
            if infected_by_A and infected_by_B:
                # if both succeeded, pick A or B at random
                if random.random() < 0.5:
                    infected_by_B = False
                else:
                    infected_by_A = False

            # --- schedule whichever infection (if any) happened ---
            if infected_by_A:
                pending_lysis.append({
                    'i': i,
                    'j': j,
                    'phage_type': 'A',
                    'steps_left': PHAGE_A_LATENT_PERIOD
                })
            elif infected_by_B:
                pending_lysis.append({
                    'i': i,
                    'j': j,
                    'phage_type': 'B',
                    'steps_left': PHAGE_B_LATENT_PERIOD
                })

def process_pending_lysis(grid, pending_lysis):
    """
    Decrement 'steps_left' each timestep; when steps_left == 0, lyse:
    - If the bacterium is still present at (i,j), remove it and release a burst
      of PHAGE_?_BURST_SIZE into grid.phages_A or grid.phages_B.
    """
    to_remove = []
    for idx, record in enumerate(pending_lysis):
        record['steps_left'] -= 1
        if record['steps_left'] <= 0:
            i, j = record['i'], record['j']
            cell = grid.grid[i][j]
            if cell is not None:
                # kill the bacterium
                grid.grid[i][j] = None
                if record['phage_type'] == 'A':
                    grid.phages_A[i, j] += PHAGE_A_BURST_SIZE
                else:
                    grid.phages_B[i, j] += PHAGE_B_BURST_SIZE
            to_remove.append(idx)

    # remove finished entries in reverse order
    for idx in reversed(to_remove):
        pending_lysis.pop(idx)

def decay_phages(grid):
    """
    Exponentially decay both phage fields each timestep.
    """
    grid.phages_A *= (1.0 - PHAGE_A_DECAY_RATE)
    grid.phages_B *= (1.0 - PHAGE_B_DECAY_RATE)
