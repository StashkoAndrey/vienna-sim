import numpy as np
from bacteria import Bacterium
import random

class SimulationGrid:
    def __init__(self, size, channel_cost, initial_nutrient=2.88):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.nutrients = np.full((size, size),
                                 fill_value=initial_nutrient,
                                 dtype=np.float32)
        self.channel_cost = channel_cost

    def place_bacterium(self, i, j, start_energy=3.0, start_uptake=1.0):
        """
        Place a single Bacterium at (i, j) if the cell is empty.
        """
        if self.grid[i][j] is None:
            self.grid[i][j] = Bacterium(
                energy=start_energy,
                uptake=start_uptake,
                channel_cost=self.channel_cost
            )

    def step(self):
        """
        Run one timestep for every bacterium: consume nutrients, pay costs,
        possibly divide, and remove dead cells.
        """
        for i in range(self.size):
            for j in range(self.size):
                bacterium = self.grid[i][j]
                if bacterium:
                    nutrient = self.nutrients[i, j]
                    bacterium.consume(nutrient)
                    # Nutrient is consumed at rate 1.0 per time step
                    self.nutrients[i, j] = max(0.0, self.nutrients[i, j] - 1.0)

                    # If energy ≤ 0, bacterium dies
                    if bacterium.energy <= 0:
                        self.grid[i][j] = None
                        continue

                    # If it can divide, place a daughter cell in a random empty neighbor
                    if bacterium.ready_to_divide():
                        neighbors = self.get_empty_neighbors(i, j)
                        if neighbors:
                            ni, nj = random.choice(neighbors)
                            self.grid[ni][nj] = bacterium.divide()

    def get_empty_neighbors(self, i, j):
        """
        Return a list of (ni, nj) tuples for empty neighbor cells.
        """
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                if self.grid[ni][nj] is None:
                    neighbors.append((ni, nj))
        return neighbors

    def count_bacteria(self):
        """
        Return the total number of living bacteria on the grid.
        """
        return sum(
            1
            for row in self.grid
            for cell in row
            if cell is not None
        )

    def total_energy(self):
        """
        Sum the energy of all living bacteria.
        """
        return sum(
            cell.energy
            for row in self.grid
            for cell in row
            if cell is not None
        )

    def total_nutrients(self):
        """
        Return the sum of nutrients across the entire grid.
        """
        return float(np.sum(self.nutrients))

    def count_channels(self):
        """
        Sum `bacterium.uptake` over all living cells.
        Interpreted as the total “number of channels” on the grid.
        """
        total = 0.0
        for row in self.grid:
            for cell in row:
                if cell is not None:
                    total += cell.uptake
        return total
