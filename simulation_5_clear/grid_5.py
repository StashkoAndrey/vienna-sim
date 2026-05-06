# grid_5.py
import random, numpy as np
from bacteria_5 import Bacterium
from phage_5 import (
    make_empty_phage_fields, attempt_new_infections,
    process_pending_lysis, decay_phages
)
from diffusion_cpp import diffuse

class SimulationGrid:
    def __init__(self, size, channel_cost, params):
        self.size = size
        self.params = params
        self.grid = [[None]*size for _ in range(size)]
        self.channel_cost = channel_cost
        self.cell_id_counter = 0

        init_nut = params["INITIAL_NUTRIENT"]
        self.nutrients_A = np.full((size, size), init_nut, dtype=np.float32)
        self.nutrients_B = np.full((size, size), init_nut, dtype=np.float32)
        self.nutrients_C = np.full((size, size), init_nut, dtype=np.float32)

        self.phages_A, self.phages_B, self.phages_C = make_empty_phage_fields(size, params)

        self.pending_lysis = []

        self.channel_time_series = {}
        self.channel_count_grid  = {}
        self.raw_index_series    = {}
        self.deviation_series    = {}
        self.raw_index_grid      = {}
        self.deviation_grid      = {}
        self.division_log        = []

    def place_bacterium(self, i, j, start_energy=None):
        if start_energy is None:
            start_energy = self.params["INITIAL_ENERGY"]
        if self.grid[i][j] is None:
            self.grid[i][j] = Bacterium(
                energy=start_energy,
                cell_id=self.cell_id_counter,
                channels_A=1,
                channels_B=0,
                channels_C=0,
                btype='A',
                params=self.params
            )
            self.cell_id_counter += 1

    def refill_nutrient_A(self, amt): self.nutrients_A[:] = np.clip(self.nutrients_A + amt, 0, self.params["INITIAL_NUTRIENT"])
    def refill_nutrient_B(self, amt): self.nutrients_B[:] = np.clip(self.nutrients_B + amt, 0, self.params["INITIAL_NUTRIENT"])
    def refill_nutrient_C(self, amt): self.nutrients_C[:] = np.clip(self.nutrients_C + amt, 0, self.params["INITIAL_NUTRIENT"])

    def step(self, t):
        process_pending_lysis(self, self.pending_lysis)
        attempt_new_infections(self, self.pending_lysis)

        upA = np.zeros((self.size, self.size), dtype=np.float32)
        upB = np.zeros_like(upA)
        upC = np.zeros_like(upA)

        self.channel_time_series[t] = []
        self.raw_index_series[t]    = []
        self.deviation_series[t]    = []

        snap_ch   = np.full((self.size, self.size), np.nan, dtype=np.float32)
        snap_idx  = np.full_like(snap_ch, np.nan)
        snap_dev  = np.full_like(snap_ch, np.nan)

        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i][j]
                if cell is None:
                    continue

                a = float(self.nutrients_A[i,j])
                b = float(self.nutrients_B[i,j])
                c = float(self.nutrients_C[i,j])
                cell.consume(a, b, c)

                upA[i,j] = min(a, cell.channels_A * self.params["UPTAKE_PER_CHANNEL"])
                upB[i,j] = min(b, cell.channels_B * self.params["UPTAKE_PER_CHANNEL"])
                upC[i,j] = min(c, cell.channels_C * self.params["UPTAKE_PER_CHANNEL"])

                tot_ch = cell.channels_A + cell.channels_B + cell.channels_C
                if tot_ch > 0:
                    idx = cell.channels_A / tot_ch
                    dev = abs(0.5 - idx)
                else:
                    idx = dev = np.nan

                self.channel_time_series[t].append(tot_ch)
                self.raw_index_series[t].append(idx)
                self.deviation_series[t].append(dev)

                snap_ch[i,j]  = tot_ch
                snap_idx[i,j] = idx
                snap_dev[i,j] = dev

                if cell.dead or cell.energy <= 0:
                    self.grid[i][j] = None
                    continue

                if cell.ready_to_divide():
                    neigh = self.get_empty_neighbors(i, j)
                    if neigh:
                        ni, nj = random.choice(neigh)
                        child = cell.divide(self.cell_id_counter)
                        self.cell_id_counter += 1
                        self.grid[ni][nj] = child

        self.nutrients_A -= upA
        self.nutrients_B -= upB
        self.nutrients_C -= upC
        decay_phages(self)

        self.channel_count_grid[t] = snap_ch
        self.raw_index_grid[t]     = snap_idx
        self.deviation_grid[t]     = snap_dev

    def get_empty_neighbors(self, i, j):
        return [
            (ni, nj)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= (ni := i + di) < self.size and 0 <= (nj := j + dj) < self.size and self.grid[ni][nj] is None
        ]

    def total_phage_A(self): return float(np.sum(self.phages_A))
    def total_phage_B(self): return float(np.sum(self.phages_B))
    def total_phage_C(self): return float(np.sum(self.phages_C))
    def total_energy(self):  return sum(cell.energy for row in self.grid for cell in row if cell)
