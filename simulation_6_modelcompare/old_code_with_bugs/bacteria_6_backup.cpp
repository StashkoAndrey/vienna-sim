#include "bacteria_6.hpp"
#include "parameters_6.hpp"
#include <random>
#include <algorithm>
#include <cmath>

// RNG setup
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dist(0.0, 1.0);
static std::normal_distribution<> norm(DIVISION_BIAS_MEAN, DIVISION_BIAS_SD);

Bacterium::Bacterium(float energy_, int cell_id, int ch_A, int ch_B, int ch_C, const std::string& btype)
    : energy(energy_), channels_A(ch_A), channels_B(ch_B), channels_C(ch_C),
      type(btype), id(cell_id), dead(false),
      channel_cost(CHANNEL_COST), maintenance_cost(MAINTENANCE_COST),
      base_metabolic_cost(BASE_METABOLIC_COST), div_threshold(DIVISION_THRESHOLD) {}

float Bacterium::build_prob(float local_nutrient, int total_channels) {
    float nutr_drive = BUILD_PROB_SLOPE * local_nutrient;
    float suppression = total_channels / (total_channels + SUPPRESSION_K);
    return std::clamp(nutr_drive * (1.0f - suppression), 0.0f, 1.0f);
}

void Bacterium::consume(float local_A, float local_B, float local_C) {
    if (dist(gen) < DEATH_PROB) {
        dead = true;
        return;
    }

    float upA = std::min(local_A, channels_A * UPTAKE_PER_CHANNEL);
    float upB = std::min(local_B, channels_B * UPTAKE_PER_CHANNEL);
    float upC = std::min(local_C, channels_C * UPTAKE_PER_CHANNEL);
    energy += (upA + upB + upC);

    int tot_ch = channels_A + channels_B + channels_C;
    energy -= (base_metabolic_cost + maintenance_cost * tot_ch + channel_cost * tot_ch);

    if (energy > 0.2f * div_threshold) {
        if (type == "A" && local_A > 0) {
            float p = build_prob(local_A, tot_ch) * std::min(local_A / INITIAL_NUTRIENT, 1.0f);
            if (dist(gen) < p) channels_A++;
        } else if (type == "B" && local_B > 0) {
            float p = build_prob(local_B, tot_ch) * std::min(local_B / INITIAL_NUTRIENT, 1.0f);
            if (dist(gen) < p) channels_B++;
        } else if (type == "C" && local_C > 0) {
            float p = build_prob(local_C, tot_ch) * std::min(local_C / INITIAL_NUTRIENT, 1.0f);
            if (dist(gen) < p) channels_C++;
        }
    }

    energy = std::max(energy, 0.0f);
    channels_A = std::max(channels_A, 0);
    channels_B = std::max(channels_B, 0);
    channels_C = std::max(channels_C, 0);
}

bool Bacterium::ready_to_divide() const {
    int tot_ch = channels_A + channels_B + channels_C;
    return !dead && energy >= div_threshold && tot_ch >= 2;
}

Bacterium* Bacterium::divide(int new_id) {
    if (dead) return nullptr;

    float r = std::clamp(static_cast<float>(norm(gen)), DIVISION_BIAS_MIN, DIVISION_BIAS_MAX);

    float child_energy = energy * (1.0f - r);
    energy *= r;

    auto split = [r](int count) -> std::pair<int, int> {
        int parent = static_cast<int>(std::round(count * r));
        return {parent, count - parent};
    };

    auto [newA, child_A] = split(channels_A);
    auto [newB, child_B] = split(channels_B);
    auto [newC, child_C] = split(channels_C);
    channels_A = newA;
    channels_B = newB;
    channels_C = newC;

    std::string new_type;
    if (dist(gen) < DIVISION_SWITCH_PROB) {
        if (type == "A") new_type = dist(gen) < 0.5 ? "B" : "C";
        else if (type == "B") new_type = dist(gen) < 0.5 ? "A" : "C";
        else new_type = dist(gen) < 0.5 ? "A" : "B";
    } else {
        new_type = type;
    }

    return new Bacterium(child_energy, new_id, child_A, child_B, child_C, new_type);
}
