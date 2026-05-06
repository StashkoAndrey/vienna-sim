#include "phage_6.hpp"
#include "parameters_6.hpp"
#include <random>
#include <cmath>
#include <algorithm>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dist(0.0, 1.0);

// Utility maps
static const std::map<std::string, float> ADSORPTION = {
    {"A", PHAGE_A_ADSORPTION_RATE}, {"B", PHAGE_B_ADSORPTION_RATE}, {"C", PHAGE_C_ADSORPTION_RATE}
};
static const std::map<std::string, int> BURST_SIZE = {
    {"A", PHAGE_A_BURST_SIZE}, {"B", PHAGE_B_BURST_SIZE}, {"C", PHAGE_C_BURST_SIZE}
};
static const std::map<std::string, float> DECAY_RATE = {
    {"A", PHAGE_A_DECAY_RATE}, {"B", PHAGE_B_DECAY_RATE}, {"C", PHAGE_C_DECAY_RATE}
};
static const std::map<std::string, int> LATENT_PERIOD = {
    {"A", PHAGE_A_LATENT_PERIOD}, {"B", PHAGE_B_LATENT_PERIOD}, {"C", PHAGE_C_LATENT_PERIOD}
};

// Local infection helper
static bool infection_attempt(float local_ph, int channels, float p_adsorb) {
    if (local_ph <= 0 || channels == 0) return false;
    float p0 = std::clamp(p_adsorb * local_ph, 0.0f, 1.0f);
    float p_infect = 1.0f - std::pow((1.0f - p0), static_cast<float>(channels));
    return dist(gen) < p_infect;
}

void attempt_new_infections(SimulationGrid& grid, std::vector<std::tuple<int, int, std::string, int>>& pending_lysis) {
    for (int i = 0; i < grid.size; ++i) {
        for (int j = 0; j < grid.size; ++j) {
            auto& cell_ptr = grid.grid[i][j];
            if (!cell_ptr) continue;

            std::map<std::string, float> local_ph = {
                {"A", grid.phages_A[i][j]},
                {"B", grid.phages_B[i][j]},
                {"C", grid.phages_C[i][j]}
            };

            std::map<std::string, int> channels = {
                {"A", cell_ptr->channels_A},
                {"B", cell_ptr->channels_B},
                {"C", cell_ptr->channels_C}
            };

            std::vector<std::string> success;
            for (const auto& [label, val] : channels) {
                if (infection_attempt(local_ph[label], val, ADSORPTION.at(label))) {
                    success.push_back(label);
                }
            }

            if (!success.empty()) {
                std::string chosen = success[std::rand() % success.size()];
                pending_lysis.emplace_back(i, j, chosen, LATENT_PERIOD.at(chosen));
            }
        }
    }
}

void process_pending_lysis(SimulationGrid& grid, std::vector<std::tuple<int, int, std::string, int>>& pending_lysis) {
    std::vector<size_t> to_remove;
    for (size_t k = 0; k < pending_lysis.size(); ++k) {
        auto& rec = pending_lysis[k];
        std::get<3>(rec)--;

        if (std::get<3>(rec) <= 0) {
            int i = std::get<0>(rec);
            int j = std::get<1>(rec);
            const std::string& ph = std::get<2>(rec);

            if (grid.grid[i][j]) {
                grid.grid[i][j].reset(); // kill cell
                if (ph == "A") grid.phages_A[i][j] += BURST_SIZE.at(ph);
                if (ph == "B") grid.phages_B[i][j] += BURST_SIZE.at(ph);
                if (ph == "C") grid.phages_C[i][j] += BURST_SIZE.at(ph);
            }

            to_remove.push_back(k);
        }
    }

    std::sort(to_remove.rbegin(), to_remove.rend());
    for (size_t idx : to_remove) {
        pending_lysis.erase(pending_lysis.begin() + idx);
    }
}

void decay_phages(SimulationGrid& grid) {
    for (int i = 0; i < grid.size; ++i) {
        for (int j = 0; j < grid.size; ++j) {
            grid.phages_A[i][j] *= (1.0f - DECAY_RATE.at("A"));
            grid.phages_B[i][j] *= (1.0f - DECAY_RATE.at("B"));
            grid.phages_C[i][j] *= (1.0f - DECAY_RATE.at("C"));
        }
    }
}
