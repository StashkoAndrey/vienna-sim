# --- grid3d.py ---
import numpy as np
from bacteria3d import Bacterium3D

class SimulationGrid3D:
    def __init__(self, size=10):
        self.size = size
        self.grid = [[[None for _ in range(size)] for _ in range(size)] for _ in range(size)]
        self.nutrients = np.ones((size, size, size)) * 5.0

    def place_bacterium(self, x, y, z):
        self.grid[z][y][x] = Bacterium3D()

    def step_bacteria(self):
        new_bacteria = []
        changed = False
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    bacterium = self.grid[z][y][x]
                    if bacterium:
                        n = self.nutrients[z, y, x]
                        bacterium.consume(n)
                        self.nutrients[z, y, x] = max(0, n - 1.0)
                        if bacterium.ready_to_divide():
                            for dz in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        if dx == dy == dz == 0:
                                            continue
                                        nx, ny, nz = x + dx, y + dy, z + dz
                                        if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                                            if self.grid[nz][ny][nx] is None:
                                                new_bacteria.append((nx, ny, nz, bacterium.divide()))
                                                changed = True
                                                break
                                    if changed: break
                                if changed: break
        for x, y, z, b in new_bacteria:
            self.grid[z][y][x] = b
        return changed