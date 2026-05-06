# --- main3d.py ---
from grid3d import SimulationGrid3D
import diffusion3d_cpp
import visualisation3d
import matplotlib.pyplot as plt

sim = SimulationGrid3D(10)
sim.place_bacterium(5, 5, 5)

max_iter = 1000
no_change = 0

for t in range(max_iter):
    changed = sim.step_bacteria()
    sim.nutrients = diffusion3d_cpp.diffuse3d(sim.nutrients, 0.1)
    visualisation3d.plot_scatter(sim.grid, t)
    no_change = 0 if changed else no_change + 1
    if no_change >= 3:
        print(f"Halted at t={t}")
        break