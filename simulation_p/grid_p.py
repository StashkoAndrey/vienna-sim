# grid_p.py

import numpy as np
import random
from bacteria_p import Bacterium
from parameters_p import (
    INITIAL_NUTRIENT,
    UPTAKE_PER_CHANNEL,
    INITIAL_ENERGY
)
from phage import (
    make_empty_phage_fields,
    attempt_new_infections,
    process_pending_lysis,
    decay_phages
)

class SimulationGrid:
    def __init__(self, size, channel_cost, initial_nutrient=INITIAL_NUTRIENT):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

        # Nutrient fields
        self.nutrients_A = np.full((size, size), initial_nutrient, dtype=np.float32)
        self.nutrients_B = np.full((size, size), initial_nutrient, dtype=np.float32)

        # ── PHAGE FIELDS ──
        self.phages_A, self.phages_B = make_empty_phage_fields(size)

        # List of pending lysis events
        self.pending_lysis = []

        self.channel_cost = channel_cost
        self.cell_id_counter = 0

        # (1) For total‐channels‐based plots / histograms:
        self.channel_time_series = {}    # { t → [ (A+B) of each living cell ] }
        self.channel_count_grid   = {}    # { t → 2D array shape=(size,size) of (A+B) or NaN }

        # ── NEW (2) FOR RAW INDEX AND DEVIATION ──
        self.raw_index_series   = {}    # { t → [ channels_A/(A+B) of each living cell ] }
        self.deviation_series   = {}    # { t → [ abs(0.5 − index) of each living cell ] }
        self.raw_index_grid     = {}    # { t → 2D array (size,size) of index = (A/(A+B)) or NaN }
        self.deviation_grid     = {}    # { t → 2D array (size,size) of dev = abs(0.5−index) or NaN }

        self.division_log = []

    def seed_phage_A(self, i, j, amount=1.0):
        self.phages_A[i, j] += amount

    def seed_phage_B(self, i, j, amount=1.0):
        self.phages_B[i, j] += amount

    def place_bacterium(self, i, j, start_energy=None):
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
        self.nutrients_A += amount
        self.nutrients_A = np.clip(self.nutrients_A, 0.0, INITIAL_NUTRIENT)

    def refill_nutrient_B(self, amount):
        self.nutrients_B += amount
        self.nutrients_B = np.clip(self.nutrients_B, 0.0, INITIAL_NUTRIENT)

    def step(self, timestep):
        """
        (0) Process pending lysis
        (1) Attempt new infections
        (2) Bacterial consumption / build / divide
        (3) Subtract uptakes
        (4) Decay phages

        Along the way, we build four new data structures per timestep:
          - channel_time_series[t]     = [ (A+B) for every living cell ]
          - raw_index_series[t]        = [ (A/(A+B)) for every living cell ]
          - deviation_series[t]        = [ |0.5 − (A/(A+B))| for every living cell ]
          - channel_count_grid[t]      = 2D array of (A+B) or NaN
          - raw_index_grid[t]          = 2D array of (A/(A+B)) or NaN
          - deviation_grid[t]          = 2D array of |0.5−index| or NaN
        """
        # Prepare empty lists/maps for this timestep
        self.channel_time_series[timestep]   = []
        self.raw_index_series[timestep]      = []
        self.deviation_series[timestep]      = []

        grid_ch_snapshot      = np.full((self.size, self.size), np.nan, dtype=np.float32)
        grid_index_snapshot   = np.full((self.size, self.size), np.nan, dtype=np.float32)
        grid_dev_snapshot     = np.full((self.size, self.size), np.nan, dtype=np.float32)

        # (0) PROCESS ANY LYSIS DUE
        process_pending_lysis(self, self.pending_lysis)

        # (1) ATTEMPT NEW INFECTIONS
        attempt_new_infections(self, self.pending_lysis)

        # (2) BACTERIAL CONSUMPTION / BUILD / DIVIDE
        uptake_map_A = np.zeros((self.size, self.size), dtype=np.float32)
        uptake_map_B = np.zeros((self.size, self.size), dtype=np.float32)

        for i in range(self.size):
            for j in range(self.size):
                bacterium = self.grid[i][j]
                if bacterium is None:
                    continue

                local_A = float(self.nutrients_A[i, j])
                local_B = float(self.nutrients_B[i, j])

                uptake_A = min(local_A, bacterium.channels_A * UPTAKE_PER_CHANNEL)
                uptake_B = min(local_B, bacterium.channels_B * UPTAKE_PER_CHANNEL)

                bacterium.consume(local_A, local_B)

                uptake_map_A[i, j] = uptake_A
                uptake_map_B[i, j] = uptake_B

                # Now that we have updated channels_A and channels_B, record them:
                total_ch = bacterium.channels_A + bacterium.channels_B
                if total_ch > 0:
                    raw_index = bacterium.channels_A / total_ch
                else:
                    raw_index = np.nan

                deviation = abs(0.5 - raw_index) if not np.isnan(raw_index) else np.nan

                # (a) total‐channels list
                self.channel_time_series[timestep].append(total_ch)
                # (b) raw‐index list
                self.raw_index_series[timestep].append(raw_index)
                # (c) deviation list
                self.deviation_series[timestep].append(deviation)

                # (d) fill the 2D snapshots
                grid_ch_snapshot[i, j]    = float(total_ch)
                grid_index_snapshot[i, j] = float(raw_index)
                grid_dev_snapshot[i, j]   = float(deviation)

                # Check death or division
                if bacterium.dead or bacterium.energy <= 0:
                    self.grid[i][j] = None
                    continue

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

                        self.division_log.append(
                            (timestep, parent_id, pre_A, pre_B, child_id, child_A, child_B)
                        )
                        self.grid[ni][nj] = offspring

        # (3) SUBTRACT UPTAKES
        self.nutrients_A -= uptake_map_A
        self.nutrients_B -= uptake_map_B

        # (4) DECAY PHAGES
        decay_phages(self)

        # Finally, store the 2D snapshots:
        self.channel_count_grid[timestep] = grid_ch_snapshot
        self.raw_index_grid[timestep]     = grid_index_snapshot
        self.deviation_grid[timestep]     = grid_dev_snapshot

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
        return float(np.sum(self.nutrients_A) + np.sum(self.nutrients_B))

    def total_phage_A(self):
        return float(np.sum(self.phages_A))

    def total_phage_B(self):
        return float(np.sum(self.phages_B))
