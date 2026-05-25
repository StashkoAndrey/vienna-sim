#include "simulation_grid_6.hpp"
#include "parameters_6.hpp"
#include "phage_6.hpp"

#include <cstdlib>
#include <algorithm>
#include <random>
#include <string>


SimulationGrid::SimulationGrid(int size_)
    : size(size_), cell_id_counter(0)
{
    grid.resize(size);
    for (int i = 0; i < size; ++i) {
        grid[i].resize(size);
    }

    initialize_nutrient_fields();
    initialize_phage_fields();
}


void SimulationGrid::initialize_nutrient_fields() {
    nutrients_A = std::vector(size, std::vector<float>(size, INITIAL_NUTRIENT));
    nutrients_B = nutrients_A;
    nutrients_C = nutrients_A;
}


void SimulationGrid::initialize_phage_fields() {
    phages_A = std::vector(size, std::vector<float>(size, INITIAL_PHAGE_A_CONCENTRATION));
    phages_B = std::vector(size, std::vector<float>(size, INITIAL_PHAGE_B_CONCENTRATION));
    phages_C = std::vector(size, std::vector<float>(size, INITIAL_PHAGE_C_CONCENTRATION));
}


void SimulationGrid::place_bacterium(int i, int j, float start_energy) {
    if (start_energy < 0) {
        start_energy = INITIAL_ENERGY;
    }

    if (!grid[i][j]) {
        grid[i][j] = std::make_unique<Bacterium>(
            start_energy,
            cell_id_counter++,
            1,
            0,
            0,
            "A"
        );
    }
}


void SimulationGrid::refill_nutrient_A(float amt) {
    for (auto& row : nutrients_A) {
        for (auto& val : row) {
            val = std::min(INITIAL_NUTRIENT, val + amt);
        }
    }
}


void SimulationGrid::refill_nutrient_B(float amt) {
    for (auto& row : nutrients_B) {
        for (auto& val : row) {
            val = std::min(INITIAL_NUTRIENT, val + amt);
        }
    }
}


void SimulationGrid::refill_nutrient_C(float amt) {
    for (auto& row : nutrients_C) {
        for (auto& val : row) {
            val = std::min(INITIAL_NUTRIENT, val + amt);
        }
    }
}


// -----------------------------------------------------------------------------
// Rapid-adaptation helper logic
// Used only when PHASE_MODEL == 2.
// Important: the actual grid/phage access happens inside SimulationGrid::step(),
// because size/grid/phages_A/B/C are class members.
// -----------------------------------------------------------------------------

static std::random_device rapid_rd;
static std::mt19937 rapid_gen(rapid_rd());
static std::uniform_real_distribution<> rapid_dist(0.0, 1.0);


static std::string choose_alternative_type(const std::string& old_type) {
    if (old_type == "A") {
        return rapid_dist(rapid_gen) < 0.5 ? "B" : "C";
    }

    if (old_type == "B") {
        return rapid_dist(rapid_gen) < 0.5 ? "A" : "C";
    }

    return rapid_dist(rapid_gen) < 0.5 ? "A" : "B";
}


static void record_rapid_transition(const std::string& old_type,
                                    const std::string& new_type,
                                    StepStats& stats) {
    if (old_type == "A" && new_type == "B") {
        stats.switch_A_to_B++;
    } else if (old_type == "A" && new_type == "C") {
        stats.switch_A_to_C++;
    } else if (old_type == "B" && new_type == "A") {
        stats.switch_B_to_A++;
    } else if (old_type == "B" && new_type == "C") {
        stats.switch_B_to_C++;
    } else if (old_type == "C" && new_type == "A") {
        stats.switch_C_to_A++;
    } else if (old_type == "C" && new_type == "B") {
        stats.switch_C_to_B++;
    }
}


// -----------------------------------------------------------------------------
// Main simulation step
// -----------------------------------------------------------------------------

void SimulationGrid::step(int /*t*/) {
    step_stats = StepStats{};

    // 0) Existing infections mature first, then new infections are attempted.
    process_pending_lysis(*this, pending_lysis, step_stats);
    attempt_new_infections(*this, pending_lysis, step_stats);

    // Model 2: rapid phage-induced adaptation.
    //
    // This is the competing model against true phase variation.
    // It does NOT create standing diversity before phage pressure.
    // A cell can change type only if matching local phage pressure is already high.
    if (PHASE_MODEL == 2) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                auto& cell_ptr = grid[i][j];

                if (!cell_ptr) {
                    continue;
                }

                float pressure = 0.0f;

                if (cell_ptr->type == "A") {
                    pressure = phages_A[i][j];
                } else if (cell_ptr->type == "B") {
                    pressure = phages_B[i][j];
                } else {
                    pressure = phages_C[i][j];
                }

                if (pressure < RAPID_ADAPTATION_PHAGE_THRESHOLD) {
                    continue;
                }

                step_stats.switch_attempts++;

                if (rapid_dist(rapid_gen) < RAPID_ADAPTATION_PROB) {
                    std::string old_type = cell_ptr->type;
                    std::string new_type = choose_alternative_type(old_type);

                    cell_ptr->type = new_type;

                    step_stats.switch_successes++;
                    record_rapid_transition(old_type, new_type, step_stats);
                } else {
                    step_stats.switch_failures++;
                }
            }
        }
    }

    std::vector<std::vector<float>> upA(size, std::vector<float>(size, 0.0f));
    std::vector<std::vector<float>> upB(size, std::vector<float>(size, 0.0f));
    std::vector<std::vector<float>> upC(size, std::vector<float>(size, 0.0f));

    // 1) Bacteria consume nutrients, die, and divide.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            auto& cell_ptr = grid[i][j];

            if (!cell_ptr) {
                continue;
            }

            float a = nutrients_A[i][j];
            float b = nutrients_B[i][j];
            float c = nutrients_C[i][j];

            // Correct uptake accounting:
            // subtract exactly what the cell actually consumed.
            UptakeResult uptake = cell_ptr->consume(a, b, c);

            upA[i][j] = uptake.A;
            upB[i][j] = uptake.B;
            upC[i][j] = uptake.C;

            if (cell_ptr->dead || cell_ptr->energy <= 0.0f) {
                cell_ptr.reset();
                continue;
            }

            if (cell_ptr->ready_to_divide()) {
                auto neigh = get_empty_neighbors(i, j);

                if (!neigh.empty()) {
                    auto [ni, nj] = neigh[std::rand() % neigh.size()];
                    auto* child = cell_ptr->divide(cell_id_counter++, step_stats);

                    if (child) {
                        grid[ni][nj] = std::unique_ptr<Bacterium>(child);
                    }
                }
            }
        }
    }

    // 2) Subtract actual uptakes and clamp to non-negative values.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            nutrients_A[i][j] = std::max(0.0f, nutrients_A[i][j] - upA[i][j]);
            nutrients_B[i][j] = std::max(0.0f, nutrients_B[i][j] - upB[i][j]);
            nutrients_C[i][j] = std::max(0.0f, nutrients_C[i][j] - upC[i][j]);
        }
    }

    // 3) Phage decay/diffusion-side update.
    decay_phages(*this);
}


std::vector<std::pair<int, int>> SimulationGrid::get_empty_neighbors(int i, int j) const {
    std::vector<std::pair<int, int>> neighbors;

    const std::vector<std::pair<int, int>> directions = {
        {-1, 0},
        {1, 0},
        {0, -1},
        {0, 1}
    };

    for (const auto& [di, dj] : directions) {
        int ni = i + di;
        int nj = j + dj;

        if (ni >= 0 && ni < size &&
            nj >= 0 && nj < size &&
            !grid[ni][nj]) {
            neighbors.emplace_back(ni, nj);
        }
    }

    return neighbors;
}


float SimulationGrid::total_phage_A() const {
    float sum = 0.0f;

    for (const auto& row : phages_A) {
        for (float val : row) {
            sum += val;
        }
    }

    return sum;
}


float SimulationGrid::total_phage_B() const {
    float sum = 0.0f;

    for (const auto& row : phages_B) {
        for (float val : row) {
            sum += val;
        }
    }

    return sum;
}


float SimulationGrid::total_phage_C() const {
    float sum = 0.0f;

    for (const auto& row : phages_C) {
        for (float val : row) {
            sum += val;
        }
    }

    return sum;
}


float SimulationGrid::total_energy() const {
    float total = 0.0f;

    for (const auto& row : grid) {
        for (const auto& cell_ptr : row) {
            if (cell_ptr) {
                total += cell_ptr->energy;
            }
        }
    }

    return total;
}