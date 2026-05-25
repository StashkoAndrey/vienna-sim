# sim6_clean changes

This bundle is a minimally invasive cleanup of your uploaded `simulation_6` C++ model.

## Biological/logic fixes

1. **Correct channel cost vs maintenance cost**
   - `MAINTENANCE_COST` is paid every timestep for existing channels.
   - `CHANNEL_COST` is paid only when a new channel is actually built.

2. **Correct nutrient uptake subtraction**
   - `Bacterium::consume()` now returns the exact A/B/C uptake used for energy gain.
   - `SimulationGrid::step()` subtracts exactly those values from nutrient fields.
   - Cells that die before consuming return zero uptake.

3. **Switch logging**
   - Per timestep: divisions, switch attempts, switch successes, switch failures.
   - Direction counts: A->B, A->C, B->A, B->C, C->A, C->B.
   - There is no invertase/failure mechanism yet, so failures should usually be 0.

4. **Phage logging**
   - Infection attempts A/B/C.
   - Established infection successes A/B/C.
   - Lysis events A/B/C.
   - Multiple pending infections on the same grid position are prevented.

5. **Phase-variant coexistence metrics**
   - `variant_richness`: how many majority variants are present at a timestep.
   - `shannon_entropy_majority`: Shannon entropy over A/B/C majority phenotypes.
   - `simpson_diversity_majority`: Simpson diversity over A/B/C majority phenotypes.
   - `minor_variant_fraction`: 1 - dominant majority-variant frequency.

## New files

- `simulation_stats_6.hpp`
- `simulation_metrics_6.hpp`
- `simulation_metrics_6.cpp`
- `analysis_6_phase_validation.py`

## Replaced files

- `bacteria_6.hpp`
- `bacteria_6.cpp`
- `simulation_grid_6.hpp`
- `simulation_grid_6.cpp`
- `phage_6.hpp`
- `phage_6.cpp`
- `main_6.cpp`

## Build note

If you use Visual Studio, add these new C++ source files to your project:

- `simulation_metrics_6.cpp`

Header-only files do not need to be compiled separately, but they must be in the include path/project folder.

If compiling manually with g++ from the simulation folder:

```bash
g++ -std=c++17 -O2 main_6.cpp simulation_grid_6.cpp bacteria_6.cpp phage_6.cpp diffusion_6.cpp simulation_metrics_6.cpp -I. -Iexternal -o sim6_clean
```

Then run:

```bash
./sim6_clean
python analysis_6_phase_validation.py
```

On Windows PowerShell with a MinGW-style compiler, the executable may be `sim6_clean.exe`.
