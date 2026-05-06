#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <tuple>
#include "external/json.hpp"

#include "parameters_6.hpp"
#include "simulation_grid_6.hpp"
#include "phage_6.hpp"
#include "bacteria_6.hpp"
#include "diffusion_6.hpp"  


// Split meal randomly with major fraction
std::tuple<float, float, float> distribute_meal(float total) {
    float major = total * MAJOR_MEAL_FRACTION;
    float remainder = total - major;
    float split = static_cast<float>(rand()) / RAND_MAX;
    float m1 = remainder * split;
    float m2 = remainder - m1;
    return {major, m1, m2};
}

nlohmann::json run_sim(int rep_idx) {
    SimulationGrid g(GRID_SIZE);
    int center = GRID_SIZE / 2;
    g.place_bacterium(center, center);

    std::vector<int> pop_A, pop_B, pop_C;
    std::vector<int> pop_major_A, pop_major_B, pop_major_C;
    std::vector<float> phA, phB, phC, energy;

    for (int step = 0; step < MAX_STEPS; ++step) {
        g.step(step);

        // Apply diffusion (currently a no-op)
        diffuse(g.nutrients_A, DIFFUSION_RATE);
        diffuse(g.nutrients_B, DIFFUSION_RATE);
        diffuse(g.nutrients_C, DIFFUSION_RATE);
        diffuse(g.phages_A, PHAGE_A_DIFFUSION_RATE);
        diffuse(g.phages_B, PHAGE_B_DIFFUSION_RATE);
        diffuse(g.phages_C, PHAGE_C_DIFFUSION_RATE);

        // Nutrient refills
        if (step > 0 && step % REFILL_INTERVAL_A == 0) {
            auto [maj, mB, mC] = distribute_meal(MEAL_TOTAL_A);
            g.refill_nutrient_A(maj);
            g.refill_nutrient_B(mB);
            g.refill_nutrient_C(mC);
        }

        if (step >= DELTA_T && (step - DELTA_T) % REFILL_INTERVAL_B == 0) {
            auto [maj, mA, mC] = distribute_meal(MEAL_TOTAL_B);
            g.refill_nutrient_B(maj);
            g.refill_nutrient_A(mA);
            g.refill_nutrient_C(mC);
        }

        if (step >= 2 * DELTA_T && (step - 2 * DELTA_T) % REFILL_INTERVAL_C == 0) {
            auto [maj, mA, mB] = distribute_meal(MEAL_TOTAL_C);
            g.refill_nutrient_C(maj);
            g.refill_nutrient_A(mA);
            g.refill_nutrient_B(mB);
        }

        // Count cell populations
        int a = 0, b = 0, c = 0;
        int majA = 0, majB = 0, majC = 0;

        for (int i = 0; i < g.size; ++i) {
            for (int j = 0; j < g.size; ++j) {
                auto& cell = g.grid[i][j];
                if (!cell) continue;

                // Label-based count
                if (cell->type == "A") ++a;
                else if (cell->type == "B") ++b;
                else ++c;

                // Majority channel
                int A = cell->channels_A;
                int B = cell->channels_B;
                int C = cell->channels_C;
                int tot = A + B + C;

                std::string maj_type = cell->type;
                if (tot > 0) {
                    if (A >= B && A >= C) maj_type = "A";
                    else if (B >= A && B >= C) maj_type = "B";
                    else maj_type = "C";
                }

                if (maj_type == "A") ++majA;
                else if (maj_type == "B") ++majB;
                else ++majC;
            }
        }

        pop_A.push_back(a);
        pop_B.push_back(b);
        pop_C.push_back(c);
        pop_major_A.push_back(majA);
        pop_major_B.push_back(majB);
        pop_major_C.push_back(majC);
        phA.push_back(g.total_phage_A());
        phB.push_back(g.total_phage_B());
        phC.push_back(g.total_phage_C());
        energy.push_back(g.total_energy());

        if (LOG_STEPS) {
            std::cout << "[rep " << rep_idx << "] t=" << step
                      << " A:" << a << " B:" << b << " C:" << c
                      << " (maj A:" << majA << " B:" << majB << " C:" << majC << ")"
                      << " E=" << energy.back()
                      << " phA=" << phA.back()
                      << " phB=" << phB.back()
                      << " phC=" << phC.back() << "\n";
        }

        if (a + b + c == 0) break;  // extinction
    }

    return {
        {"pop_A", pop_A},
        {"pop_B", pop_B},
        {"pop_C", pop_C},
        {"pop_major_A", pop_major_A},
        {"pop_major_B", pop_major_B},
        {"pop_major_C", pop_major_C},
        {"phA", phA},
        {"phB", phB},
        {"phC", phC},
        {"energy", energy}
    };
}

int main() {
    std::vector<nlohmann::json> all_results;

    for (int r = 0; r < REPLICATES; ++r) {
        std::cout << "\n⚑ Starting replicate " << r + 1 << " / " << REPLICATES << "\n";
        all_results.push_back(run_sim(r));
    }

    // Aggregate results
    nlohmann::json agg;
    for (int i = 0; i < REPLICATES; ++i) {
        const auto& r = all_results[i];
        for (auto& [key, val] : r.items()) {
            agg["all_" + key].push_back(val);
        }
    }

    std::ofstream out("simulation_results.json");
    out << agg.dump(2);
    out.close();

    std::cout << "✓ Simulations complete → simulation_results.json\n";
    return 0;
}
