# grid_s3.py
import numpy as np
import random
from bacteria_s3 import Bacterium
from parameters_s3 import (
    INITIAL_NUTRIENT,
    UPTAKE_PER_CHANNEL,
    INITIAL_ENERGY,
)

class SimulationGrid:
    def __init__(self, size, channel_cost, initial_nutrient=INITIAL_NUTRIENT):
        self.size = size
        # The 2D grid of Bacterium objects (or None if empty)
        self.grid = [[None for _ in range(size)] for _ in range(size)]

        # Two nutrient fields (A and B), initialized everywhere to `initial_nutrient`
        self.nutrients_A = np.full((size, size), fill_value=initial_nutrient, dtype=np.float32)
        self.nutrients_B = np.full((size, size), fill_value=initial_nutrient, dtype=np.float32)

        self.channel_cost = channel_cost
        self.cell_id_counter = 0

        # ────────────────────────────────────────────────────────────────────────
        # Original bookkeeping:
        #    - channel_time_series[t]: list of (channels_A+channels_B) for each live cell at t.
        #    - division_log: list of tuples (step, parent_id, pre_A, pre_B, child_id, child_A, child_B).
        # ────────────────────────────────────────────────────────────────────────
        self.channel_time_series = {}
        self.division_log = []

        # ────────────────────────────────────────────────────────────────────────
        # NEW: At each timestep t, we will record:
        #    1) index_time_series[t]:     a Python list of raw_index ∈ [0,1] for every cell
        #    2) deviation_time_series[t]: a Python list of |ideal − raw_index| for every cell
        #
        #    3) index_grid[t]:     a NumPy array of shape (size, size) whose (i,j) entry
        #       is raw_index for the cell at (i,j), or NaN if empty.
        #    4) deviation_grid[t]: a NumPy array of shape (size, size) whose (i,j) entry
        #       is |ideal − raw_index| for the cell at (i,j), or NaN if empty.
        # ────────────────────────────────────────────────────────────────────────
        self.index_time_series = {}
        self.deviation_time_series = {}
        self.index_grid = {}      # maps t → 2D np.ndarray of raw_index (NaN for empties)
        self.deviation_grid = {}  # maps t → 2D np.ndarray of deviation (NaN for empties)

    def place_bacterium(self, i, j, start_energy=None):
        """
        Place a new bacterium at (i, j) if empty, with 1 A‐channel and 0 B‐channels by default.
        """
        if start_energy is None:
            start_energy = INITIAL_ENERGY
        if self.grid[i][j] is None:
            self.grid[i][j] = Bacterium(
                energy=start_energy,
                cell_id=self.cell_id_counter,
                channels_A=1,
                channels_B=0
            )
            self.cell_id_counter += 1

    def refill_nutrient_A(self, amount):
        """
        Add `amount` to nutrient A everywhere, then clamp to [0, INITIAL_NUTRIENT].
        """
        self.nutrients_A += amount
        self.nutrients_A = np.clip(self.nutrients_A, 0.0, INITIAL_NUTRIENT)

    def refill_nutrient_B(self, amount):
        """
        Add `amount` to nutrient B everywhere, then clamp to [0, INITIAL_NUTRIENT].
        """
        self.nutrients_B += amount
        self.nutrients_B = np.clip(self.nutrients_B, 0.0, INITIAL_NUTRIENT)

    def step(self, timestep):
        """
        Perform one simulation timestep:
          1) Each live bacterium consumes local A & B, pays costs, builds channels, may die.
          2) We subtract actual uptake from each nutrient field.
          3) We record channel‐counts (A+B), plus (new) raw_index and deviation for analysis.
          4) Cells above the division threshold attempt to divide into an empty neighbor.
        """
        # ────────────────────────────────────────────────────────────────────────
        # Initialize per‐timestep containers:
        #   .channel_time_series[t]:     list of total_channels for each cell
        #   .index_time_series[t]:       list of raw_index for each cell
        #   .deviation_time_series[t]:   list of |ideal−raw_index| for each cell
        #
        #   .index_grid[t]:     2D array of raw_index (shape=(size,size)), initialized NaN
        #   .deviation_grid[t]: 2D array of deviation (shape=(size,size)), initialized NaN
        # ────────────────────────────────────────────────────────────────────────
        self.channel_time_series[timestep]   = []
        self.index_time_series[timestep]     = []
        self.deviation_time_series[timestep] = []
        self.index_grid[timestep]     = np.full((self.size, self.size), np.nan, dtype=np.float32)
        self.deviation_grid[timestep] = np.full((self.size, self.size), np.nan, dtype=np.float32)

        for i in range(self.size):
            for j in range(self.size):
                bacterium = self.grid[i][j]
                if bacterium is None:
                    # Leave index_grid[timestep][i,j] = NaN,
                    # and deviation_grid[timestep][i,j] = NaN
                    continue

                # 1) Local nutrient concentrations
                local_A = float(self.nutrients_A[i, j])
                local_B = float(self.nutrients_B[i, j])

                # 2) Compute how much can be taken (per‐channel uptake limit)
                uptake_A = min(local_A, bacterium.channels_A * UPTAKE_PER_CHANNEL)
                uptake_B = min(local_B, bacterium.channels_B * UPTAKE_PER_CHANNEL)

                # 3) Actual consumption (this updates bacterium.energy, possibly builds channels)
                bacterium.consume(local_A, local_B)

                # 4) Subtract the actual uptakes from nutrient fields
                self.nutrients_A[i, j] = local_A - uptake_A
                self.nutrients_B[i, j] = local_B - uptake_B

                # 5) Record total channels (A+B)
                total_ch = bacterium.channels_A + bacterium.channels_B
                self.channel_time_series[timestep].append(total_ch)

                # ─────────── NEW: COMPUTE “raw_index” AND “deviation” ───────────
                if total_ch > 0:
                    raw_index = bacterium.channels_A / float(total_ch)
                else:
                    # If a cell somehow has zero channels, assign 0.5
                    raw_index = 0.5

                # The “ideal” depends on cell.type: A‐types want index=1.0; B‐types want index=0.0
                if bacterium.type == 'A':
                    ideal = 1.0
                else:
                    ideal = 0.0

                deviation = abs(ideal - raw_index)

                # Append to the per‐timestep lists
                self.index_time_series[timestep].append(raw_index)
                self.deviation_time_series[timestep].append(deviation)

                # Store into the 2D grid arrays at (i, j) only
                self.index_grid[timestep][i, j]     = raw_index
                self.deviation_grid[timestep][i, j] = deviation
                # ────────────────────────────────────────────────────────────────

                # 6) If the cell’s energy dropped to ≤ 0.0 during consume, it dies now
                if bacterium.energy <= 0.0:
                    self.grid[i][j] = None
                    continue

                # 7) Attempt division if the cell is ready
                if bacterium.ready_to_divide():
                    neighbors = self.get_empty_neighbors(i, j)
                    if neighbors:
                        ni, nj = random.choice(neighbors)

                        parent_id = bacterium.id
                        pre_A = bacterium.channels_A
                        pre_B = bacterium.channels_B

                        offspring = bacterium.divide(self.cell_id_counter)
                        child_id = offspring.id
                        child_A = offspring.channels_A
                        child_B = offspring.channels_B
                        self.cell_id_counter += 1

                        # Log the division event
                        self.division_log.append(
                            (timestep, parent_id, pre_A, pre_B, child_id, child_A, child_B)
                        )

                        # Place the new child in the chosen empty spot
                        self.grid[ni][nj] = offspring

    def get_empty_neighbors(self, i, j):
        """
        Return a list of coordinates (ni, nj) that are orthogonally adjacent to (i, j)
        and are currently empty. Used to place a dividing daughter cell.
        """
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni][nj] is None:
                neighbors.append((ni, nj))
        return neighbors

    def count_bacteria(self):
        """
        Return the total number of live cells on the grid.
        """
        return sum(1 for row in self.grid for cell in row if cell is not None)

    def total_energy(self):
        """
        Return the sum of energies of all living cells on the grid.
        """
        return sum(cell.energy for row in self.grid for cell in row if cell)

    def total_nutrients(self):
        """
        Return the sum of both nutrient fields (A + B) over the entire grid.
        """
        return float(np.sum(self.nutrients_A) + np.sum(self.nutrients_B))
