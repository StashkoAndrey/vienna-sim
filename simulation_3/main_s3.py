# main_s3.py
import logging
import pickle

from grid_s3 import SimulationGrid
from diffusion_cpp import diffuse
from parameters_s3 import (
    GRID_SIZE,
    MAX_STEPS,
    DIFFUSION_RATE,
    REPLICATES,
    REFILL_INTERVAL_A,
    REFILL_INTERVAL_B,
    MEAL_TOTAL_A,
    MEAL_TOTAL_B,
    CHANNEL_COST,
)

#logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def run_simulation(channel_cost):
    """
    Returns nine items:
      1) pop_A_series:        list (≤ MAX_STEPS)  = # of A‐type cells at each timestep
      2) pop_B_series:        list (≤ MAX_STEPS)  = # of B‐type cells at each timestep
      3) chan_Atype_series:   dict { t: [channels_A of each A‐type cell at t] }
      4) chan_Btype_series:   dict { t: [channels_B of each B‐type cell at t] }
      5) energy_series:       list (≤ MAX_STEPS)  = total energy of all cells at t
      6) index_series:        dict { t: [raw_index ∈ [0,1] of every cell at t] }
      7) deviation_series:    dict { t: [|ideal − raw_index| for every cell at t] }
      8) index_grid:          dict { t: 2D np.ndarray of raw_index for each (i,j) }
      9) deviation_grid:      dict { t: 2D np.ndarray of deviation     for each (i,j) }
    """
    grid = SimulationGrid(GRID_SIZE, channel_cost)
    center = GRID_SIZE // 2
    grid.place_bacterium(center, center)

    pop_A_series = []
    pop_B_series = []
    energy_series = []
    chan_Atype_series = {}
    chan_Btype_series = {}
    index_series = {}       # NEW: lists of raw_index per cell
    deviation_series = {}   # NEW: lists of |ideal−index| per cell

    for step in range(MAX_STEPS):
        chan_Atype_series[step] = []
        chan_Btype_series[step] = []
        # grid.step(step) will automatically fill:
        #   grid.index_time_series[step], grid.deviation_time_series[step],
        #   grid.index_grid[step], grid.deviation_grid[step].
        grid.step(step)

        # ------------- tally channel counts & population counts -------------
        count_A_cells = 0
        count_B_cells = 0

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = grid.grid[i][j]
                if cell is None:
                    continue

                if cell.type == 'A':
                    count_A_cells += 1
                    chan_Atype_series[step].append(cell.channels_A)
                else:  # cell.type == 'B'
                    count_B_cells += 1
                    chan_Btype_series[step].append(cell.channels_B)

        pop_A_series.append(count_A_cells)
        pop_B_series.append(count_B_cells)

        # ------------- diffuse nutrients -------------
        grid.nutrients_A = diffuse(grid.nutrients_A, DIFFUSION_RATE)
        grid.nutrients_B = diffuse(grid.nutrients_B, DIFFUSION_RATE)

        # ------------- refill nutrients if needed -------------
        if step % REFILL_INTERVAL_A == 0 and step != 0:
            grid.refill_nutrient_A(MEAL_TOTAL_A)
        if step % REFILL_INTERVAL_B == 0 and step != 0:
            grid.refill_nutrient_B(MEAL_TOTAL_B)

        # ------------- record total energy -------------
        total_energy = grid.total_energy()
        energy_series.append(total_energy)

        # ------------- copy out index & deviation lists -------------
        index_series[step]     = list(grid.index_time_series[step])
        deviation_series[step] = list(grid.deviation_time_series[step])

        #logging.info(
        #    f"Step {step}: A_cells={count_A_cells}, B_cells={count_B_cells}, "
        #    f"TotalEnergy={total_energy:.2f}"
        #)

        # ------------- early stop if no cells remain -------------
        if (count_A_cells + count_B_cells) == 0:
            break

    # Return everything (including the new grids)
    return (
        pop_A_series,
        pop_B_series,
        chan_Atype_series,
        chan_Btype_series,
        energy_series,
        index_series,
        deviation_series,
        grid.index_grid,
        grid.deviation_grid
    )


def main():
    all_pop_A         = []
    all_pop_B         = []
    all_chan_Atype    = []
    all_chan_Btype    = []
    all_energy        = []
    all_index         = []  # NEW: list of dicts {t: [raw_index per cell]}
    all_deviation     = []  # NEW: list of dicts {t: [|ideal−index| per cell]}
    all_index_grid    = []  # NEW: list of dicts {t: 2D raw_index grid}
    all_deviation_grid= []  # NEW: list of dicts {t: 2D deviation grid}

    for _ in range(REPLICATES):
        (
            series_Acells,
            series_Bcells,
            chanA,
            chanB,
            e_series,
            idx_series,
            dev_series,
            idx_grid_dict,
            dev_grid_dict
        ) = run_simulation(CHANNEL_COST)

        all_pop_A.append(series_Acells)
        all_pop_B.append(series_Bcells)
        all_chan_Atype.append(chanA)
        all_chan_Btype.append(chanB)
        all_energy.append(e_series)
        all_index.append(idx_series)
        all_deviation.append(dev_series)
        all_index_grid.append(idx_grid_dict)
        all_deviation_grid.append(dev_grid_dict)

    # ────────────────────────────────────────────────────────────────────────────
    # 1) Write the full data (with index/deviation/grids) to simulation_data_with_index.pkl
    # ────────────────────────────────────────────────────────────────────────────
    full_results = {
        'all_pop_A':         all_pop_A,
        'all_pop_B':         all_pop_B,
        'all_chan_Atype':    all_chan_Atype,
        'all_chan_Btype':    all_chan_Btype,
        'all_energy':        all_energy,
        'all_index':         all_index,
        'all_deviation':     all_deviation,
        'all_index_grid':    all_index_grid,
        'all_deviation_grid':all_deviation_grid
    }

    with open('simulation_data_with_index.pkl', 'wb') as f:
        pickle.dump(full_results, f)
    print("Wrote full data → simulation_data_with_index.pkl")

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Also write legacy data (pop, channels, energy only) to simulation_data.pkl
    # ────────────────────────────────────────────────────────────────────────────
    legacy_results = {
        'all_pop_A':      all_pop_A,
        'all_pop_B':      all_pop_B,
        'all_chan_Atype': all_chan_Atype,
        'all_chan_Btype': all_chan_Btype,
        'all_energy':     all_energy
    }

    with open('simulation_data.pkl', 'wb') as f2:
        pickle.dump(legacy_results, f2)
    print("Wrote legacy data → simulation_data.pkl")


if __name__ == '__main__':
    main()
