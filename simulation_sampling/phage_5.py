# Updated phage_5.py
import random
import numpy as np

def make_empty_phage_fields(size, param_dict):
    phA = np.full((size,size), param_dict['INITIAL_PHAGE_A_CONCENTRATION'], dtype=np.float32)
    phB = np.full((size,size), param_dict['INITIAL_PHAGE_B_CONCENTRATION'], dtype=np.float32)
    phC = np.full((size,size), param_dict['INITIAL_PHAGE_C_CONCENTRATION'], dtype=np.float32)
    return phA, phB, phC

def _infection_attempt(local_ph, channels, p_adsorb):
    if local_ph <= 0 or channels == 0:
        return False
    p0 = p_adsorb * local_ph
    p0 = max(0.0, min(1.0, p0))
    p_infect = 1.0 - (1.0 - p0) ** channels
    return random.random() < p_infect

def attempt_new_infections(grid, pending_lysis):
    param = grid.param_dict
    adsorp = {
        'A': param['PHAGE_A_ADSORPTION_RATE'],
        'B': param['PHAGE_B_ADSORPTION_RATE'],
        'C': param['PHAGE_C_ADSORPTION_RATE']
    }
    latent = {
        'A': param['PHAGE_A_LATENT_PERIOD'],
        'B': param['PHAGE_B_LATENT_PERIOD'],
        'C': param['PHAGE_C_LATENT_PERIOD']
    }
    size = grid.size
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
            channels = {'A': cell.channels_A, 'B': cell.channels_B, 'C': cell.channels_C}
            successes = [
                ph for ph in ('A','B','C')
                if _infection_attempt(local[ph], channels[ph], adsorp[ph])
            ]
            if successes:
                ph = random.choice(successes)
                pending_lysis.append({
                    'i': i, 'j': j,
                    'phage_type': ph,
                    'steps_left': latent[ph]
                })

def process_pending_lysis(grid, pending_lysis):
    burst = {
        'A': grid.param_dict['PHAGE_A_BURST_SIZE'],
        'B': grid.param_dict['PHAGE_B_BURST_SIZE'],
        'C': grid.param_dict['PHAGE_C_BURST_SIZE']
    }
    kill = []
    for idx, rec in enumerate(pending_lysis):
        rec['steps_left'] -= 1
        if rec['steps_left'] <= 0:
            i, j, ph = rec['i'], rec['j'], rec['phage_type']
            if grid.grid[i][j] is not None:
                grid.grid[i][j] = None
                getattr(grid, f'phages_{ph}')[i, j] += burst[ph]
            kill.append(idx)
    for idx in reversed(kill):
        pending_lysis.pop(idx)

def decay_phages(grid):
    grid.phages_A *= (1.0 - grid.param_dict['PHAGE_A_DECAY_RATE'])
    grid.phages_B *= (1.0 - grid.param_dict['PHAGE_B_DECAY_RATE'])
    grid.phages_C *= (1.0 - grid.param_dict['PHAGE_C_DECAY_RATE'])
