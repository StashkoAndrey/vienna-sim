# phage_5.py
import random
import numpy as np

def make_empty_phage_fields(size, params):
    """Return three phage fields, each seeded uniformly."""
    phA = np.full((size, size), params["INITIAL_PHAGE_A_CONCENTRATION"], dtype=np.float32)
    phB = np.full((size, size), params["INITIAL_PHAGE_B_CONCENTRATION"], dtype=np.float32)
    phC = np.full((size, size), params["INITIAL_PHAGE_C_CONCENTRATION"], dtype=np.float32)
    return phA, phB, phC

def _infection_attempt(local_ph, channels, p_adsorb):
    """Independent adsorption of each channel ⇒ Beta-binomial closed form."""
    if local_ph <= 0 or channels == 0:
        return False
    p0 = p_adsorb * local_ph
    p0 = max(0.0, min(1.0, p0))
    p_infect = 1.0 - (1.0 - p0) ** channels
    return random.random() < p_infect

def attempt_new_infections(grid, pending_lysis):
    size = grid.size
    p = grid.params

    ADSORPTION = {
        'A': p["PHAGE_A_ADSORPTION_RATE"],
        'B': p["PHAGE_B_ADSORPTION_RATE"],
        'C': p["PHAGE_C_ADSORPTION_RATE"]
    }
    LATENT = {
        'A': p["PHAGE_A_LATENT_PERIOD"],
        'B': p["PHAGE_B_LATENT_PERIOD"],
        'C': p["PHAGE_C_LATENT_PERIOD"]
    }

    for i in range(size):
        for j in range(size):
            cell = grid.grid[i][j]
            if cell is None:
                continue

            local = {
                'A': float(grid.phages_A[i,j]),
                'B': float(grid.phages_B[i,j]),
                'C': float(grid.phages_C[i,j])
            }
            channels = {
                'A': cell.channels_A,
                'B': cell.channels_B,
                'C': cell.channels_C
            }
            successes = [
                ph for ph in ('A', 'B', 'C')
                if _infection_attempt(local[ph], channels[ph], ADSORPTION[ph])
            ]
            if successes:
                ph = random.choice(successes)
                pending_lysis.append({
                    'i': i,
                    'j': j,
                    'phage_type': ph,
                    'steps_left': LATENT[ph]
                })

def process_pending_lysis(grid, pending_lysis):
    p = grid.params
    BURST_SIZE = {
        'A': p["PHAGE_A_BURST_SIZE"],
        'B': p["PHAGE_B_BURST_SIZE"],
        'C': p["PHAGE_C_BURST_SIZE"]
    }

    kill = []
    for idx, rec in enumerate(pending_lysis):
        rec['steps_left'] -= 1
        if rec['steps_left'] <= 0:
            i, j, ph = rec['i'], rec['j'], rec['phage_type']
            if grid.grid[i][j] is not None:
                grid.grid[i][j] = None
                getattr(grid, f'phages_{ph}')[i, j] += BURST_SIZE[ph]
            kill.append(idx)
    for idx in reversed(kill):
        pending_lysis.pop(idx)

def decay_phages(grid):
    p = grid.params
    DECAY_RATE = {
        'A': p["PHAGE_A_DECAY_RATE"],
        'B': p["PHAGE_B_DECAY_RATE"],
        'C': p["PHAGE_C_DECAY_RATE"]
    }

    grid.phages_A *= (1.0 - DECAY_RATE['A'])
    grid.phages_B *= (1.0 - DECAY_RATE['B'])
    grid.phages_C *= (1.0 - DECAY_RATE['C'])
