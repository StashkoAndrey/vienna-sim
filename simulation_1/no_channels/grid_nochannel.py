import numpy as np
from bacteria_nochannel import Bacterium
import random

class SimulationGrid:
    def __init__(self, size, initial_nutrient=2.0):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.nutrients = np.full((size, size), fill_value=initial_nutrient, dtype=np.float32)

    def place_bacterium(self, i, j):
        if self.grid[i][j] is None:
            self.grid[i][j] = Bacterium()

    def step(self):
        new_bacteria = []
        for i in range(self.size):
            for j in range(self.size):
                bacterium = self.grid[i][j]
                if bacterium:
                    nutrient = self.nutrients[i, j]
                    bacterium.consume(nutrient)
                    self.nutrients[i, j] = max(0.0, self.nutrients[i, j] - 1.0)
                    if bacterium.energy <= 0:
                        self.grid[i][j] = None
                        continue
                    if bacterium.ready_to_divide():
                        neighbors = self.get_empty_neighbors(i, j)
                        if neighbors:
                            ni, nj = random.choice(neighbors)
                            self.grid[ni][nj] = bacterium.divide()

    def get_empty_neighbors(self, i, j):
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                if self.grid[ni][nj] is None:
                    neighbors.append((ni, nj))
        return neighbors

    def count_bacteria(self):
        return sum(1 for row in self.grid for cell in row if cell is not None)

    def total_energy(self):
        return sum(cell.energy for row in self.grid for cell in row if cell is not None)

    def total_nutrients(self):
        return np.sum(self.nutrients)
