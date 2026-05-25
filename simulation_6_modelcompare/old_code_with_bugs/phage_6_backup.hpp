#ifndef PHAGE_HPP
#define PHAGE_HPP

#include "simulation_grid_6.hpp"
#include <tuple>
#include <string>

// Lysis record: i, j, type, steps_left
struct LysisRecord {
    int i, j;
    std::string phage_type;
    int steps_left;
};

// Attempt to infect viable cells
void attempt_new_infections(SimulationGrid& grid, std::vector<std::tuple<int, int, std::string, int>>& pending_lysis);

// Process lysis events that have matured
void process_pending_lysis(SimulationGrid& grid, std::vector<std::tuple<int, int, std::string, int>>& pending_lysis);

// Apply per-timestep phage decay
void decay_phages(SimulationGrid& grid);

#endif // PHAGE_HPP
