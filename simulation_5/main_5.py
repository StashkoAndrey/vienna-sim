# main_5.py
#
# 3-species bacterial simulation with phages, randomised nutrient
# pulses, detailed metrics, and NEW majority-by-channel population
# counters. Also saves snapshots for animation.
# ---------------------------------------------------------------------
import logging
import random
import pickle
import numpy as np

from grid_5 import SimulationGrid
from diffusion_cpp import diffuse
from parameters_extracted_5 import (
    # grid / run
    GRID_SIZE, MAX_STEPS, REPLICATES, CHANNEL_COST,
    # diffusion
    DIFFUSION_RATE,
    PHAGE_A_DIFFUSION_RATE, PHAGE_B_DIFFUSION_RATE, PHAGE_C_DIFFUSION_RATE,
    # meal schedule
    REFILL_INTERVAL_A, REFILL_INTERVAL_B, REFILL_INTERVAL_C,
    DELTA_T,
    MEAL_TOTAL_A, MEAL_TOTAL_B, MEAL_TOTAL_C,
    # random split
    MAJOR_MEAL_FRACTION,
    # logging
    LOG_STEPS
)

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s',
                    force=True)

# ───────────────────────── helper: split one meal ─────────────────────
def _distribute_meal(total, major_frac=MAJOR_MEAL_FRACTION):
    major = total * major_frac
    remainder = total - major
    split = random.random()
    m1 = remainder * split
    m2 = remainder - m1
    return major, m1, m2

# ───────────────────────── run one replicate ──────────────────────────
def run_sim(rep_idx, channel_cost=CHANNEL_COST):
    g = SimulationGrid(GRID_SIZE, channel_cost)
    centre = GRID_SIZE // 2
    g.place_bacterium(centre, centre)

    # global time-series (label-based populations, phages, energy)
    pop_A, pop_B, pop_C = [], [], []
    phA,   phB,   phC   = [], [], []
    energy = []

    # NEW majority-by-channel series
    pop_major_A, pop_major_B, pop_major_C = [], [], []

    # per-timestep detail dicts
    index_dic, dev_dic, H_dic = {}, {}, {}
    chanA_dic, chanB_dic, chanC_dic = {}, {}, {}

    # ─── snapshots for animation ───
    snapshots = {'channels': [], 'phage_A': [], 'phage_B': [], 'phage_C': []}

    for step in range(MAX_STEPS):
        g.step(step)

        # diffusion
        g.nutrients_A = diffuse(g.nutrients_A, DIFFUSION_RATE)
        g.nutrients_B = diffuse(g.nutrients_B, DIFFUSION_RATE)
        g.nutrients_C = diffuse(g.nutrients_C, DIFFUSION_RATE)
        g.phages_A    = diffuse(g.phages_A,  PHAGE_A_DIFFUSION_RATE)
        g.phages_B    = diffuse(g.phages_B,  PHAGE_B_DIFFUSION_RATE)
        g.phages_C    = diffuse(g.phages_C,  PHAGE_C_DIFFUSION_RATE)

        # nutrient refills
        if step and step % REFILL_INTERVAL_A == 0:
            maj, mB, mC = _distribute_meal(MEAL_TOTAL_A)
            g.refill_nutrient_A(maj)
            g.refill_nutrient_B(mB)
            g.refill_nutrient_C(mC)

        if step and step >= DELTA_T and (step - DELTA_T) % REFILL_INTERVAL_B == 0:
            maj, mA, mC = _distribute_meal(MEAL_TOTAL_B)
            g.refill_nutrient_B(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_C(mC)

        if step and step >= 2 * DELTA_T and (step - 2 * DELTA_T) % REFILL_INTERVAL_C == 0:
            maj, mA, mB = _distribute_meal(MEAL_TOTAL_C)
            g.refill_nutrient_C(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_B(mB)

        # per-cell stats
        a = b = c = 0
        majA = majB = majC = 0
        IA_lst, DEV_lst, H_lst = [], [], []
        Achan, Bchan, Cchan = [], [], []

        # frame snapshot
        snap_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = g.grid[i][j]
                if cell is None:
                    snap_grid[i, j] = {'A': 0, 'B': 0, 'C': 0}
                    continue

                # label-based
                if cell.type == 'A': a += 1
                elif cell.type == 'B': b += 1
                else: c += 1

                # majority type
                tot_ch = cell.channels_A + cell.channels_B + cell.channels_C
                if tot_ch == 0:
                    maj_type = cell.type
                else:
                    if (cell.channels_A >= cell.channels_B and cell.channels_A >= cell.channels_C):
                        maj_type = 'A'
                    elif (cell.channels_B >= cell.channels_A and cell.channels_B >= cell.channels_C):
                        maj_type = 'B'
                    else:
                        maj_type = 'C'
                if maj_type == 'A': majA += 1
                elif maj_type == 'B': majB += 1
                else: majC += 1

                # entropy
                if tot_ch:
                    pA = cell.channels_A / tot_ch
                    pB = cell.channels_B / tot_ch
                    pC = cell.channels_C / tot_ch
                    IA = pA
                    DEV = abs(0.5 - IA)
                    H = -sum(p*np.log2(p) for p in (pA, pB, pC) if p > 0)
                    IA_lst.append(IA)
                    DEV_lst.append(DEV)
                    H_lst.append(H)

                # label-based channel counts
                if cell.type == 'A' and cell.channels_A: Achan.append(cell.channels_A)
                if cell.type == 'B' and cell.channels_B: Bchan.append(cell.channels_B)
                if cell.type == 'C' and cell.channels_C: Cchan.append(cell.channels_C)

                # store channel snapshot
                snap_grid[i, j] = {
                    'A': cell.channels_A,
                    'B': cell.channels_B,
                    'C': cell.channels_C,
                }

        # save per-timestep data
        index_dic[step] = IA_lst
        dev_dic[step]   = DEV_lst
        H_dic[step]     = H_lst
        chanA_dic[step] = Achan
        chanB_dic[step] = Bchan
        chanC_dic[step] = Cchan

        pop_A.append(a); pop_B.append(b); pop_C.append(c)
        pop_major_A.append(majA); pop_major_B.append(majB); pop_major_C.append(majC)
        phA.append(g.total_phage_A()); phB.append(g.total_phage_B()); phC.append(g.total_phage_C())
        energy.append(g.total_energy())

        snapshots['channels'].append(snap_grid)
        snapshots['phage_A'].append(g.phages_A.copy())
        snapshots['phage_B'].append(g.phages_B.copy())
        snapshots['phage_C'].append(g.phages_C.copy())

        if LOG_STEPS:
                 logging.info(f"[rep {rep_idx}] t={step:4d}  A:{a:3d} B:{b:3d} C:{c:3d} "
                 f"(maj A:{majA} B:{majB} C:{majC}) "
                 f"E={energy[-1]:7.1f}  phA={phA[-1]:.0f} phB={phB[-1]:.0f} phC={phC[-1]:.0f}")


        if a + b + c == 0:
            break  # extinction

    # save snapshots for first replicate
    if rep_idx == 0:
        with open('snapshot_run0.pkl', 'wb') as f:
            pickle.dump(snapshots, f)

    return dict(
        pop_A=pop_A, pop_B=pop_B, pop_C=pop_C,
        pop_major_A=pop_major_A, pop_major_B=pop_major_B, pop_major_C=pop_major_C,
        phA=phA, phB=phB, phC=phC, energy=energy,
        index=index_dic, deviation=dev_dic, entropy=H_dic,
        chan_Atype=chanA_dic, chan_Btype=chanB_dic, chan_Ctype=chanC_dic
    )

# ────────────────────────── run all replicates ─────────────────────────
def main():
    all_runs = []
    for r in range(REPLICATES):
        logging.info(f"⚑  starting replicate {r+1}/{REPLICATES}")
        all_runs.append(run_sim(r))

    agg = {
        'all_pop_A': [run['pop_A'] for run in all_runs],
        'all_pop_B': [run['pop_B'] for run in all_runs],
        'all_pop_C': [run['pop_C'] for run in all_runs],
        'pop_major_A': [run['pop_major_A'] for run in all_runs],
        'pop_major_B': [run['pop_major_B'] for run in all_runs],
        'pop_major_C': [run['pop_major_C'] for run in all_runs],
        'all_phage_A': [run['phA'] for run in all_runs],
        'all_phage_B': [run['phB'] for run in all_runs],
        'all_phage_C': [run['phC'] for run in all_runs],
        'all_energy':  [run['energy']  for run in all_runs],
        'all_index':     [run['index']     for run in all_runs],
        'all_deviation': [run['deviation'] for run in all_runs],
        'all_entropy':   [run['entropy']   for run in all_runs],
        'all_chan_Atype': [run['chan_Atype'] for run in all_runs],
        'all_chan_Btype': [run['chan_Btype'] for run in all_runs],
        'all_chan_Ctype': [run['chan_Ctype'] for run in all_runs],
    }

    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(agg, f)

    logging.info("✓ simulations complete → simulation_results.pkl")

if __name__ == '__main__':
    main()
