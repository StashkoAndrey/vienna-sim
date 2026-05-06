import numpy as np
import random
from bacteria_s21 import Bacterium
from parameters import INITIAL_NUTRIENT, UPTAKE_PER_CHANNEL, INITIAL_ENERGY

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
            self.grid[i][j] = Bacterium(energy=start_energy, cell_id=self.cell_id_counter)
            self.cell_id_counter += 1

    def refill_nutrients(self, amount_per_cell):
        # explicit refill, clamp to initial
        self.nutrients += amount_per_cell
        self.nutrients = np.clip(self.nutrients, 0.0, INITIAL_NUTRIENT)

    def step(self, timestep):
        self.channel_time_series[timestep] = []
        for i in range(self.size):
            for j in range(self.size):
                bacterium = self.grid[i][j]
                if bacterium is None:
                    continue

                # nutrient uptake
                local_nutrient = float(self.nutrients[i, j])
                uptake_amount = min(local_nutrient, bacterium.channels * UPTAKE_PER_CHANNEL)
                bacterium.consume(uptake_amount, local_nutrient)
                self.nutrients[i, j] = local_nutrient - uptake_amount
                self.channel_time_series[timestep].append(bacterium.channels)

                # death check
                if bacterium.energy <= 0:
                    self.grid[i][j] = None
                    continue

                # division
                if bacterium.ready_to_divide():
                    neighbors = self.get_empty_neighbors(i, j)
                    if neighbors:
                        ni, nj = random.choice(neighbors)
                        # log pre-division
                        parent_id = bacterium.id
                        pre_split = bacterium.channels
                        # perform division
                        offspring = bacterium.divide(self.cell_id_counter)
                        child_id = offspring.id
                        child_chans = offspring.channels
                        self.cell_id_counter += 1
                        # record split
                        self.division_log.append((timestep, parent_id, pre_split, child_id, child_chans))
                        # place offspring
                        self.grid[ni][nj] = offspring



    def get_empty_neighbors(self, i, j):
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni][nj] is None:
                neighbors.append((ni, nj))
        return neighbors

    def count_bacteria(self):
        return sum(1 for row in self.grid for cell in row if cell is not None)

    def total_energy(self):
        return sum(cell.energy for row in self.grid for cell in row if cell)

    def total_nutrients(self):
        return float(np.sum(self.nutrients))
