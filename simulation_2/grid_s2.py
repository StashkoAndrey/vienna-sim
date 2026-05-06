import numpy as np
import random
from bacteria_s21 import Bacterium
from parameters import INITIAL_NUTRIENT, INITIAL_ENERGY, UPTAKE_PER_CHANNEL

class SimulationGrid:
    def __init__(self, size, channel_cost, initial_nutrient=INITIAL_NUTRIENT):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.nutrients = np.full((size, size), fill_value=initial_nutrient, dtype=np.float32)
        self.channel_cost = channel_cost
        self.cell_id_counter = 0
        self.channel_time_series = {}
        self.division_log = []  # records (step, parent_id, pre_split_channels, child_id, child_channels)

    def place_bacterium(self, i, j, start_energy=None):
        if start_energy is None:
            start_energy = INITIAL_ENERGY
        if self.grid[i][j] is None:
            # start with one channel by default
            self.grid[i][j] = Bacterium(energy=start_energy, cell_id=self.cell_id_counter, channels=1)
            self.cell_id_counter += 1

    def step(self, timestep):
        """
        Advance simulation by one time step:
        1) Uptake and metabolism
        2) Death
        3) Division
        4) Record post-event channel counts
        """
        # Snapshot existing cells to avoid double-processing offspring
        coords = [(i, j) for i in range(self.size) for j in range(self.size)
                  if self.grid[i][j] is not None]

        for i, j in coords:
            bacterium = self.grid[i][j]
            if bacterium is None:
                continue

            # 1) Uptake & metabolism
            local_nutrient = float(self.nutrients[i, j])
            uptake = min(local_nutrient, bacterium.channels * UPTAKE_PER_CHANNEL)
            bacterium.consume(uptake, local_nutrient)
            self.nutrients[i, j] = local_nutrient - uptake

            if timestep < 5:    # just for the first few steps
                print(f"[DEBUG] Step {timestep:2d} at ({i},{j}) channels={bacterium.channels} "
                    f"local_nutrient={local_nutrient:.2f} uptake={uptake:.2f}")

            # 2) Death
            if bacterium.energy <= 0:
                self.grid[i][j] = None
                continue

            # 3) Division
            if bacterium.ready_to_divide():
                empty = self.get_empty_neighbors(i, j)
                if empty:
                    ni, nj = random.choice(empty)
                    parent_id = bacterium.id
                    pre_split = bacterium.channels
                    offspring = bacterium.divide(self.cell_id_counter)
                    child_id = offspring.id
                    child_chans = offspring.channels
                    self.cell_id_counter += 1
                    self.division_log.append((timestep, parent_id, pre_split, child_id, child_chans))
                    self.grid[ni][nj] = offspring

        # 4) Record channel counts after all events
        self.channel_time_series[timestep] = [
            cell.channels
            for row in self.grid
            for cell in row if cell is not None
        ]

    def refill_nutrients(self, amount_total):
        """
        Distribute **amount_total** nutrients evenly across the grid,
        capping each cell at INITIAL_NUTRIENT.
        """
        per_cell = amount_total / (self.size ** 2)
        self.nutrients += per_cell
        self.nutrients = np.clip(self.nutrients, 0.0, INITIAL_NUTRIENT)

    def get_empty_neighbors(self, i, j):
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni][nj] is None:
                neighbors.append((ni, nj))
        return neighbors

    def count_bacteria(self):
        return sum(1 for row in self.grid for cell in row if cell is not None)

    def total_energy(self):
        return sum(cell.energy for row in self.grid for cell in row if cell)

    def total_nutrients(self):
        return float(np.sum(self.nutrients))
