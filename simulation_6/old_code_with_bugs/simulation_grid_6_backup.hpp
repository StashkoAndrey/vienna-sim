#ifndef SIMULATION_GRID_HPP
#define SIMULATION_GRID_HPP

#include <vector>
#include <map>
#include <memory>
#include <string>
#include "bacteria_6.hpp"

class SimulationGrid {
public:
    SimulationGrid(int size);

    void place_bacterium(int i, int j, float start_energy = -1.0f);
    void refill_nutrient_A(float amt);
    void refill_nutrient_B(float amt);
    void refill_nutrient_C(float amt);
    void step(int t);

    // Convenience summaries
    float total_phage_A() const;
    float total_phage_B() const;
    float total_phage_C() const;
    float total_energy() const;

    // Grid data
    int size;
    std::vector<std::vector<std::unique_ptr<Bacterium>>> grid;

    // Nutrient & phage fields
    std::vector<std::vector<float>> nutrients_A;
    std::vector<std::vector<float>> nutrients_B;
    std::vector<std::vector<float>> nutrients_C;
    std::vector<std::vector<float>> phages_A;
    std::vector<std::vector<float>> phages_B;
    std::vector<std::vector<float>> phages_C;

    // Output and state
    int cell_id_counter;
    std::vector<std::map<int, float>> channel_time_series;
    std::vector<std::map<int, float>> raw_index_series;
    std::vector<std::map<int, float>> deviation_series;

private:
    std::vector<std::tuple<int, int, std::string, int>> pending_lysis;

    std::vector<std::pair<int, int>> get_empty_neighbors(int i, int j) const;

    void initialize_phage_fields();
    void initialize_nutrient_fields();
};

#endif // SIMULATION_GRID_HPP
