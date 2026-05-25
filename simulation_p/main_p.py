# main_p.py

import logging
import pickle
import numpy as np
import random
from grid_p import SimulationGrid
from diffusion_cpp import diffuse
from parameters_p import (
    GRID_SIZE,
    MAX_STEPS,
    DIFFUSION_RATE,
    REPLICATES,
    REFILL_INTERVAL_A,
    REFILL_INTERVAL_B,
    MEAL_TOTAL_A,
    MEAL_TOTAL_B,
    CHANNEL_COST,
    PHAGE_A_DIFFUSION_RATE,
    PHAGE_B_DIFFUSION_RATE,
    DELTA_T
)

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def run_simulation(channel_cost):
    """
    Runs one replicate of the simulation. 
    Now returns, in addition to previous outputs:
      - raw_index_replicate   (dict: t -> [A/(A+B) for each cell])
      - deviation_replicate   (dict: t -> [abs(0.5 - A/(A+B)) for each cell])
      - raw_index_grid_repl   (dict: t -> 2D array of raw_index or NaN)
      - deviation_grid_repl   (dict: t -> 2D array of deviation or NaN)
    """
    grid = SimulationGrid(GRID_SIZE, channel_cost)
    center = GRID_SIZE // 2
    grid.place_bacterium(center, center)

    # -- keep the old series as before --
    pop_A_series      = []
    pop_B_series      = []
    chan_Atype_series = {}
    chan_Btype_series = {}
    energy_series     = []
    phage_A_series    = []
    phage_B_series    = []

    # -- NEW: placeholders for index/deviation --
    raw_index_replicate   = {}  # { t -> [raw_index of cells] }
    deviation_replicate   = {}  # { t -> [deviation of cells] }
    raw_index_grid_repl   = {}  # { t -> 2D np.ndarray of raw_index (NaN if empty) }
    deviation_grid_repl   = {}  # { t -> 2D np.ndarray of dev (NaN if empty) }

    for step in range(MAX_STEPS):
        chan_Atype_series[step] = []
        chan_Btype_series[step] = []
        raw_index_replicate[step]   = []
        deviation_replicate[step]   = []
        raw_index_grid_repl[step]   = None
        deviation_grid_repl[step]   = None

        # (1) INFECTION / LYSIS + BACTERIAL CONSUMPTION / BUILD / DIVIDE
        grid.step(step)

        # (2) TALLY A‐ and B‐type populations & channels
        count_A_cells = 0
        count_B_cells = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = grid.grid[i][j]
                if cell is None:
                    continue
                total_ch = cell.channels_A + cell.channels_B

                # raw‐index + deviation for this one cell
                if total_ch > 0:
                    idx = cell.channels_A / total_ch
                    dev = abs(0.5 - idx)
                else:
                    idx = np.nan
                    dev = np.nan

                raw_index_replicate[step].append(idx)
                deviation_replicate[step].append(dev)

                if cell.type == 'A':
                    count_A_cells += 1
                    chan_Atype_series[step].append(cell.channels_A)
                else:
                    count_B_cells += 1
                    chan_Btype_series[step].append(cell.channels_B)

        pop_A_series.append(count_A_cells)
        pop_B_series.append(count_B_cells)

        # (3) DIFFUSE nutrient fields
        grid.nutrients_A = diffuse(grid.nutrients_A, DIFFUSION_RATE)
        grid.nutrients_B = diffuse(grid.nutrients_B, DIFFUSION_RATE)

        # (4) DIFFUSE phage fields
        grid.phages_A = diffuse(grid.phages_A, PHAGE_A_DIFFUSION_RATE)
        grid.phages_B = diffuse(grid.phages_B, PHAGE_B_DIFFUSION_RATE)

        # (5) REFILL nutrients on staggered schedules
        if step != 0 and (step % REFILL_INTERVAL_A) == 0:
            grid.refill_nutrient_A(MEAL_TOTAL_A)
        if step != 0 and step >= DELTA_T and ((step - DELTA_T) % REFILL_INTERVAL_B) == 0:
            grid.refill_nutrient_B(MEAL_TOTAL_B)

        # (6) RECORD total energy & total phage A/B
        total_energy = grid.total_energy()
        energy_series.append(total_energy)

        total_phA = grid.total_phage_A()
        total_phB = grid.total_phage_B()
        phage_A_series.append(total_phA)
        phage_B_series.append(total_phB)

        #logging.info(
        #    f"Step {step}: A_cells={count_A_cells}, B_cells={count_B_cells}, "
        #    f"Energy={total_energy:.2f}, PhageA={total_phA:.1f}, PhageB={total_phB:.1f}"
        #)

        # Now capture the 2D index & deviation grids from `grid`:
        raw_index_grid_repl[step] = grid.raw_index_grid[step]
        deviation_grid_repl[step] = grid.deviation_grid[step]

        if (count_A_cells + count_B_cells) == 0:
            break

    return {
        'pop_A_series':          pop_A_series,
        'pop_B_series':          pop_B_series,
        'chan_Atype_series':     chan_Atype_series,
        'chan_Btype_series':     chan_Btype_series,
        'energy_series':         energy_series,
        'phage_A_series':        phage_A_series,
        'phage_B_series':        phage_B_series,
        'channel_ts_replicate':  grid.channel_time_series,
        'channel_grid_replicate':grid.channel_count_grid,

        # ── THESE FOUR FIELDS ARE NEW ──
        'all_index':             raw_index_replicate,
        'all_deviation':         deviation_replicate,
        'all_index_grid':        raw_index_grid_repl,
        'all_deviation_grid':    deviation_grid_repl
    }

def main():
    all_pop_A           = []
    all_pop_B           = []
    all_chan_A          = []
    all_chan_B          = []
    all_energy          = []
    all_phage_A         = []
    all_phage_B         = []
    all_channel_ts      = []
    all_channel_grids   = []
    all_index_list      = []
    all_deviation_list  = []
    all_index_grids     = []
    all_deviation_grids = []

    for _ in range(REPLICATES):
        result = run_simulation(CHANNEL_COST)
        all_pop_A.append(result['pop_A_series'])
        all_pop_B.append(result['pop_B_series'])
        all_chan_A.append(result['chan_Atype_series'])
        all_chan_B.append(result['chan_Btype_series'])
        all_energy.append(result['energy_series'])
        all_phage_A.append(result['phage_A_series'])
        all_phage_B.append(result['phage_B_series'])
        all_channel_ts.append(result['channel_ts_replicate'])
        all_channel_grids.append(result['channel_grid_replicate'])

        # ── COLLECT THE FOUR NEW FIELDS ──
        all_index_list.append(result['all_index'])
        all_deviation_list.append(result['all_deviation'])
        all_index_grids.append(result['all_index_grid'])
        all_deviation_grids.append(result['all_deviation_grid'])

    aggregated = {
        'all_pop_A':           all_pop_A,
        'all_pop_B':           all_pop_B,
        'all_chan_Atype':      all_chan_A,
        'all_chan_Btype':      all_chan_B,
        'all_energy':          all_energy,
        'all_phage_A':         all_phage_A,
        'all_phage_B':         all_phage_B,
        'all_channel_ts':      all_channel_ts,
        'all_channel_grids':   all_channel_grids,

        # ── WRITE THESE EXACTLY SO `analysis_p.py` CAN FIND THEM ──
        'all_index':           all_index_list,
        'all_deviation':       all_deviation_list,
        'all_index_grid':      all_index_grids,
        'all_deviation_grid':  all_deviation_grids
    }

    with open('simulation_results_p.pkl', 'wb') as f:
        pickle.dump(aggregated, f)

    print("Simulation complete -> simulation_results_p.pkl")

if __name__ == '__main__':
    main()
