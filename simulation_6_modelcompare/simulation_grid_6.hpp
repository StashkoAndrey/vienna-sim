#ifndef SIMULATION_GRID_HPP
#define SIMULATION_GRID_HPP

#include <vector>
#include <memory>
#include <string>
#include <tuple>

#include "bacteria_6.hpp"
#include "simulation_stats_6.hpp"

class SimulationGrid {
public:
    explicit SimulationGrid(int size);

    void place_bacterium(int i, int j, float start_energy = -1.0f);
    void refill_nutrient_A(float amt);
    void refill_nutrient_B(float amt);
    void refill_nutrient_C(float amt);
    void step(int t);

    float total_phage_A() const;
    float total_phage_B() const;
    float total_phage_C() const;
    float total_energy() const;

    int size;
    std::vector<std::vector<std::unique_ptr<Bacterium>>> grid;

    std::vector<std::vector<float>> nutrients_A;
    std::vector<std::vector<float>> nutrients_B;
    std::vector<std::vector<float>> nutrients_C;
    std::vector<std::vector<float>> phages_A;
    std::vector<std::vector<float>> phages_B;
    std::vector<std::vector<float>> phages_C;

    int cell_id_counter;

    // Event counters from the most recent step. These are reset at the start of step().
    StepStats step_stats;

private:
    // i, j, phage_type, steps_left
    std::vector<std::tuple<int, int, std::string, int>> pending_lysis;

    std::vector<std::pair<int, int>> get_empty_neighbors(int i, int j) const;

    void initialize_phage_fields();
    void initialize_nutrient_fields();
};

#endif // SIMULATION_GRID_HPP
