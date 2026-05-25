#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <tuple>
#include <cstdlib>

#include "external/json.hpp"

#include "parameters_6.hpp"
#include "simulation_grid_6.hpp"
#include "simulation_metrics_6.hpp"
#include "diffusion_6.hpp"

std::tuple<float, float, float> distribute_meal(float total) {
    float major = total * MAJOR_MEAL_FRACTION;
    float remainder = total - major;
    float split = static_cast<float>(std::rand()) / RAND_MAX;
    float m1 = remainder * split;
    float m2 = remainder - m1;
    return {major, m1, m2};
}

template <typename T>
void push_metric(std::vector<T>& v, T value) {
    v.push_back(value);
}

nlohmann::json run_sim(int rep_idx) {
    SimulationGrid g(GRID_SIZE);
    int center = GRID_SIZE / 2;
    g.place_bacterium(center, center);

    std::vector<int> pop_A, pop_B, pop_C, total_population;
    std::vector<int> pop_major_A, pop_major_B, pop_major_C;
    std::vector<int> variant_richness;

    std::vector<float> phA, phB, phC, energy;
    std::vector<float> nutrient_A, nutrient_B, nutrient_C;
    std::vector<float> shannon_entropy_majority;
    std::vector<float> simpson_diversity_majority;
    std::vector<float> minor_variant_fraction;
    std::vector<float> mean_channels_total, mean_channels_A, mean_channels_B, mean_channels_C;

    std::vector<int> divisions;
    std::vector<int> switch_attempts, switch_successes, switch_failures;
    std::vector<int> switch_A_to_B, switch_A_to_C, switch_B_to_A, switch_B_to_C, switch_C_to_A, switch_C_to_B;

    std::vector<int> infection_attempts_A, infection_attempts_B, infection_attempts_C;
    std::vector<int> infection_successes_A, infection_successes_B, infection_successes_C;
    std::vector<int> lysis_events_A, lysis_events_B, lysis_events_C;

    for (int step = 0; step < MAX_STEPS; ++step) {
        g.step(step);

        diffuse(g.nutrients_A, DIFFUSION_RATE);
        diffuse(g.nutrients_B, DIFFUSION_RATE);
        diffuse(g.nutrients_C, DIFFUSION_RATE);
        diffuse(g.phages_A, PHAGE_A_DIFFUSION_RATE);
        diffuse(g.phages_B, PHAGE_B_DIFFUSION_RATE);
        diffuse(g.phages_C, PHAGE_C_DIFFUSION_RATE);

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

        StepStats m = compute_step_metrics(g, g.step_stats);

        push_metric(pop_A, m.pop_A);
        push_metric(pop_B, m.pop_B);
        push_metric(pop_C, m.pop_C);
        push_metric(total_population, m.total_population);

        push_metric(pop_major_A, m.pop_major_A);
        push_metric(pop_major_B, m.pop_major_B);
        push_metric(pop_major_C, m.pop_major_C);
        push_metric(variant_richness, m.variant_richness);

        push_metric(phA, m.total_phage_A);
        push_metric(phB, m.total_phage_B);
        push_metric(phC, m.total_phage_C);
        push_metric(energy, m.total_energy);
        push_metric(nutrient_A, m.total_nutrient_A);
        push_metric(nutrient_B, m.total_nutrient_B);
        push_metric(nutrient_C, m.total_nutrient_C);

        push_metric(shannon_entropy_majority, m.shannon_entropy_majority);
        push_metric(simpson_diversity_majority, m.simpson_diversity_majority);
        push_metric(minor_variant_fraction, m.minor_variant_fraction);
        push_metric(mean_channels_total, m.mean_channels_total);
        push_metric(mean_channels_A, m.mean_channels_A);
        push_metric(mean_channels_B, m.mean_channels_B);
        push_metric(mean_channels_C, m.mean_channels_C);

        push_metric(divisions, m.divisions);
        push_metric(switch_attempts, m.switch_attempts);
        push_metric(switch_successes, m.switch_successes);
        push_metric(switch_failures, m.switch_failures);
        push_metric(switch_A_to_B, m.switch_A_to_B);
        push_metric(switch_A_to_C, m.switch_A_to_C);
        push_metric(switch_B_to_A, m.switch_B_to_A);
        push_metric(switch_B_to_C, m.switch_B_to_C);
        push_metric(switch_C_to_A, m.switch_C_to_A);
        push_metric(switch_C_to_B, m.switch_C_to_B);

        push_metric(infection_attempts_A, m.infection_attempts_A);
        push_metric(infection_attempts_B, m.infection_attempts_B);
        push_metric(infection_attempts_C, m.infection_attempts_C);
        push_metric(infection_successes_A, m.infection_successes_A);
        push_metric(infection_successes_B, m.infection_successes_B);
        push_metric(infection_successes_C, m.infection_successes_C);
        push_metric(lysis_events_A, m.lysis_events_A);
        push_metric(lysis_events_B, m.lysis_events_B);
        push_metric(lysis_events_C, m.lysis_events_C);

        if (LOG_STEPS) {
            std::cout << "[rep " << rep_idx << "] t=" << step
                      << " pop=" << m.total_population
                      << " A:" << m.pop_A << " B:" << m.pop_B << " C:" << m.pop_C
                      << " maj(A/B/C):" << m.pop_major_A << "/" << m.pop_major_B << "/" << m.pop_major_C
                      << " H=" << m.shannon_entropy_majority
                      << " minor=" << m.minor_variant_fraction
                      << " switches=" << m.switch_successes << "/" << m.switch_attempts
                      << " ph=" << (m.total_phage_A + m.total_phage_B + m.total_phage_C)
                      << "\n";
        }

        if (m.total_population == 0) break;
    }

    return {
        {"pop_A", pop_A},
        {"pop_B", pop_B},
        {"pop_C", pop_C},
        {"total_population", total_population},
        {"pop_major_A", pop_major_A},
        {"pop_major_B", pop_major_B},
        {"pop_major_C", pop_major_C},
        {"variant_richness", variant_richness},
        {"phA", phA},
        {"phB", phB},
        {"phC", phC},
        {"energy", energy},
        {"nutrient_A", nutrient_A},
        {"nutrient_B", nutrient_B},
        {"nutrient_C", nutrient_C},
        {"shannon_entropy_majority", shannon_entropy_majority},
        {"simpson_diversity_majority", simpson_diversity_majority},
        {"minor_variant_fraction", minor_variant_fraction},
        {"mean_channels_total", mean_channels_total},
        {"mean_channels_A", mean_channels_A},
        {"mean_channels_B", mean_channels_B},
        {"mean_channels_C", mean_channels_C},
        {"divisions", divisions},
        {"switch_attempts", switch_attempts},
        {"switch_successes", switch_successes},
        {"switch_failures", switch_failures},
        {"switch_A_to_B", switch_A_to_B},
        {"switch_A_to_C", switch_A_to_C},
        {"switch_B_to_A", switch_B_to_A},
        {"switch_B_to_C", switch_B_to_C},
        {"switch_C_to_A", switch_C_to_A},
        {"switch_C_to_B", switch_C_to_B},
        {"infection_attempts_A", infection_attempts_A},
        {"infection_attempts_B", infection_attempts_B},
        {"infection_attempts_C", infection_attempts_C},
        {"infection_successes_A", infection_successes_A},
        {"infection_successes_B", infection_successes_B},
        {"infection_successes_C", infection_successes_C},
        {"lysis_events_A", lysis_events_A},
        {"lysis_events_B", lysis_events_B},
        {"lysis_events_C", lysis_events_C}
    };
}

int main() {
    std::vector<nlohmann::json> all_results;

    for (int r = 0; r < REPLICATES; ++r) {
        std::cout << "\nStarting replicate " << r + 1 << " / " << REPLICATES << "\n";
        all_results.push_back(run_sim(r));
    }

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

    std::cout << "Simulations complete -> simulation_results.json\n";
    return 0;
}
