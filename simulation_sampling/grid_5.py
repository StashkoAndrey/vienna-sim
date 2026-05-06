# Updated grid_5.py
import random, numpy as np
from bacteria_5 import Bacterium
from phage_5 import make_empty_phage_fields, attempt_new_infections, process_pending_lysis, decay_phages
from diffusion_cpp import diffuse

class SimulationGrid:
    def __init__(self, size, param_dict):
        self.size = size
        self.grid = [[None]*size for _ in range(size)]

        self.param_dict = param_dict

        self.invertase_penalty = param_dict.get("INERTASE_PENALTY", 0.0)
        self.switch_prob       = param_dict.get("DIVISION_SWITCH_PROB", 0.0)

        ini_nutr = param_dict['INITIAL_NUTRIENT']
        self.nutrients_A = np.full((size, size), ini_nutr, dtype=np.float32)
        self.nutrients_B = np.full((size, size), ini_nutr, dtype=np.float32)
        self.nutrients_C = np.full((size, size), ini_nutr, dtype=np.float32)

        self.phages_A, self.phages_B, self.phages_C = make_empty_phage_fields(size, param_dict)

        self.pending_lysis   = []
        self.channel_cost    = param_dict['CHANNEL_COST']
        self.cell_id_counter = 0

    def place_bacterium(self, i, j, start_energy=None):
        if start_energy is None:
            start_energy = self.param_dict['INITIAL_ENERGY']
        if self.grid[i][j] is None:
            self.grid[i][j] = Bacterium(
                energy=start_energy, cell_id=self.cell_id_counter,
                channels_A=1, channels_B=0, channels_C=0, btype='A',
                invertase_penalty=self.invertase_penalty,
                switch_prob=self.switch_prob,
                param_dict=self.param_dict
            )
            self.cell_id_counter += 1

    def refill_nutrient_A(self, amt): self.nutrients_A[:] = np.clip(self.nutrients_A+amt, 0, self.param_dict['INITIAL_NUTRIENT'])
    def refill_nutrient_B(self, amt): self.nutrients_B[:] = np.clip(self.nutrients_B+amt, 0, self.param_dict['INITIAL_NUTRIENT'])
    def refill_nutrient_C(self, amt): self.nutrients_C[:] = np.clip(self.nutrients_C+amt, 0, self.param_dict['INITIAL_NUTRIENT'])

    def step(self, t):
        process_pending_lysis(self, self.pending_lysis)
        attempt_new_infections(self, self.pending_lysis)

        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i][j]
                if cell is None:
                    continue

                a = float(self.nutrients_A[i,j]); b = float(self.nutrients_B[i,j]); c = float(self.nutrients_C[i,j])
                cell.consume(a,b,c)

                if cell.dead or cell.energy<=0:
                    self.grid[i][j]=None
                    continue
                if cell.ready_to_divide():
                    neigh = self.get_empty_neighbors(i,j)
                    if neigh:
                        ni,nj = random.choice(neigh)
                        child = cell.divide(self.cell_id_counter)
                        self.cell_id_counter+=1
                        self.grid[ni][nj]=child

        self.nutrients_A = np.clip(self.nutrients_A, 0, None)
        self.nutrients_B = np.clip(self.nutrients_B, 0, None)
        self.nutrients_C = np.clip(self.nutrients_C, 0, None)
        decay_phages(self)

    def get_empty_neighbors(self,i,j):
        neigh=[]
        for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
            ni,nj=i+di,j+dj
            if 0<=ni<self.size and 0<=nj<self.size and self.grid[ni][nj] is None:
                neigh.append((ni,nj))
        return neigh

    def total_phage_A(self): return float(np.clip(np.sum(self.phages_A), 0, None))
    def total_phage_B(self): return float(np.clip(np.sum(self.phages_B), 0, None))
    def total_phage_C(self): return float(np.clip(np.sum(self.phages_C), 0, None))
    def total_energy(self):  return float(sum(max(cell.energy, 0.0) for row in self.grid for cell in row if cell))
