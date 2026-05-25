#include "simulation_metrics_6.hpp"
#include "simulation_grid_6.hpp"

#include <algorithm>
#include <cmath>
#include <string>

static float sum_field(const std::vector<std::vector<float>>& field) {
    float total = 0.0f;
    for (const auto& row : field) {
        for (float v : row) total += v;
    }
    return total;
}

static void add_entropy_terms(int count, int total, float& shannon, float& simpson_sum) {
    if (count <= 0 || total <= 0) return;
    float p = static_cast<float>(count) / static_cast<float>(total);
    shannon -= p * (std::log(p) / std::log(2.0f));
    simpson_sum += p * p;
}

StepStats compute_step_metrics(const SimulationGrid& grid, const StepStats& event_stats) {
    StepStats out = event_stats;

    float sum_ch_total = 0.0f;
    float sum_ch_A = 0.0f;
    float sum_ch_B = 0.0f;
    float sum_ch_C = 0.0f;

    for (int i = 0; i < grid.size; ++i) {
        for (int j = 0; j < grid.size; ++j) {
            const auto& cell = grid.grid[i][j];
            if (!cell) continue;

            out.total_population++;

            if (cell->type == "A") out.pop_A++;
            else if (cell->type == "B") out.pop_B++;
            else out.pop_C++;

            int A = cell->channels_A;
            int B = cell->channels_B;
            int C = cell->channels_C;
            int total_channels = A + B + C;

            sum_ch_A += A;
            sum_ch_B += B;
            sum_ch_C += C;
            sum_ch_total += total_channels;

            std::string majority = cell->type;
            if (total_channels > 0) {
                if (A >= B && A >= C) majority = "A";
                else if (B >= A && B >= C) majority = "B";
                else majority = "C";
            }

            if (majority == "A") out.pop_major_A++;
            else if (majority == "B") out.pop_major_B++;
            else out.pop_major_C++;
        }
    }

    if (out.total_population > 0) {
        out.mean_channels_total = sum_ch_total / out.total_population;
        out.mean_channels_A = sum_ch_A / out.total_population;
        out.mean_channels_B = sum_ch_B / out.total_population;
        out.mean_channels_C = sum_ch_C / out.total_population;

        int max_major = std::max({out.pop_major_A, out.pop_major_B, out.pop_major_C});
        out.minor_variant_fraction = 1.0f - static_cast<float>(max_major) / out.total_population;

        out.variant_richness = 0;
        if (out.pop_major_A > 0) out.variant_richness++;
        if (out.pop_major_B > 0) out.variant_richness++;
        if (out.pop_major_C > 0) out.variant_richness++;

        float shannon = 0.0f;
        float simpson_sum = 0.0f;
        add_entropy_terms(out.pop_major_A, out.total_population, shannon, simpson_sum);
        add_entropy_terms(out.pop_major_B, out.total_population, shannon, simpson_sum);
        add_entropy_terms(out.pop_major_C, out.total_population, shannon, simpson_sum);
        out.shannon_entropy_majority = shannon;
        out.simpson_diversity_majority = 1.0f - simpson_sum;
    }

    out.total_nutrient_A = sum_field(grid.nutrients_A);
    out.total_nutrient_B = sum_field(grid.nutrients_B);
    out.total_nutrient_C = sum_field(grid.nutrients_C);

    out.total_phage_A = grid.total_phage_A();
    out.total_phage_B = grid.total_phage_B();
    out.total_phage_C = grid.total_phage_C();

    out.total_energy = grid.total_energy();

    return out;
}
