#ifndef SIMULATION_METRICS_6_HPP
#define SIMULATION_METRICS_6_HPP

#include "simulation_stats_6.hpp"

class SimulationGrid;

// Combines event stats from the current timestep with population/phenotype/system totals.
StepStats compute_step_metrics(const SimulationGrid& grid, const StepStats& event_stats);

#endif // SIMULATION_METRICS_6_HPP
