#ifndef PHAGE_HPP
#define PHAGE_HPP

#include "simulation_grid_6.hpp"
#include "simulation_stats_6.hpp"

#include <tuple>
#include <string>
#include <vector>

void attempt_new_infections(SimulationGrid& grid,
                            std::vector<std::tuple<int, int, std::string, int>>& pending_lysis,
                            StepStats& stats);

void process_pending_lysis(SimulationGrid& grid,
                           std::vector<std::tuple<int, int, std::string, int>>& pending_lysis,
                           StepStats& stats);

void decay_phages(SimulationGrid& grid);

#endif // PHAGE_HPP
