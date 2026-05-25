#include "bacteria_6.hpp"
#include "parameters_6.hpp"

#include <random>
#include <algorithm>
#include <cmath>
#include <string>


static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dist(0.0, 1.0);
static std::normal_distribution<> norm(DIVISION_BIAS_MEAN, DIVISION_BIAS_SD);


Bacterium::Bacterium(float energy_,
                     int cell_id,
                     int ch_A,
                     int ch_B,
                     int ch_C,
                     const std::string& btype)
    : energy(energy_),
      channels_A(ch_A),
      channels_B(ch_B),
      channels_C(ch_C),
      type(btype),
      id(cell_id),
      dead(false),
      channel_cost(CHANNEL_COST),
      maintenance_cost(MAINTENANCE_COST),
      base_metabolic_cost(BASE_METABOLIC_COST),
      div_threshold(DIVISION_THRESHOLD)
{
}


float Bacterium::build_prob(float local_nutrient, int total_channels) {
    float nutr_drive = BUILD_PROB_SLOPE * local_nutrient;

    float suppression = static_cast<float>(total_channels) /
                        (static_cast<float>(total_channels) + SUPPRESSION_K);

    return std::clamp(nutr_drive * (1.0f - suppression), 0.0f, 1.0f);
}


bool Bacterium::try_build_channel(char channel_type,
                                  float local_nutrient,
                                  int total_channels) {
    if (local_nutrient <= 0.0f) {
        return false;
    }

    float p = build_prob(local_nutrient, total_channels) *
              std::min(local_nutrient / INITIAL_NUTRIENT, 1.0f);

    if (dist(gen) >= p) {
        return false;
    }

    // CHANNEL_COST is a one-time construction cost.
    if (energy < channel_cost) {
        return false;
    }

    if (channel_type == 'A') {
        channels_A++;
    } else if (channel_type == 'B') {
        channels_B++;
    } else if (channel_type == 'C') {
        channels_C++;
    } else {
        return false;
    }

    energy -= channel_cost;
    return true;
}


UptakeResult Bacterium::consume(float local_A,
                                float local_B,
                                float local_C) {
    UptakeResult uptake;

    // If the cell dies before metabolism, it should not consume nutrients.
    if (dist(gen) < DEATH_PROB) {
        dead = true;
        return uptake;
    }

    // Uptake is calculated using channels that existed at the start
    // of the timestep.
    uptake.A = std::min(local_A, channels_A * UPTAKE_PER_CHANNEL);
    uptake.B = std::min(local_B, channels_B * UPTAKE_PER_CHANNEL);
    uptake.C = std::min(local_C, channels_C * UPTAKE_PER_CHANNEL);

    energy += uptake.A + uptake.B + uptake.C;

    int total_channels_before = channels_A + channels_B + channels_C;

    // Recurring cost:
    // BASE_METABOLIC_COST is always paid.
    // MAINTENANCE_COST is paid per existing channel.
    // CHANNEL_COST is not paid here; it is paid only when a channel is built.
    energy -= base_metabolic_cost +
              maintenance_cost * static_cast<float>(total_channels_before);

    energy = std::max(energy, 0.0f);

    // Channel construction happens after uptake/maintenance.
    // A newly built channel does not contribute to this timestep's uptake.
    if (energy > 0.2f * div_threshold) {
        int total_channels_now = channels_A + channels_B + channels_C;

        if (type == "A") {
            try_build_channel('A', local_A, total_channels_now);
        } else if (type == "B") {
            try_build_channel('B', local_B, total_channels_now);
        } else if (type == "C") {
            try_build_channel('C', local_C, total_channels_now);
        }
    }

    energy = std::max(energy, 0.0f);

    channels_A = std::max(channels_A, 0);
    channels_B = std::max(channels_B, 0);
    channels_C = std::max(channels_C, 0);

    return uptake;
}


bool Bacterium::ready_to_divide() const {
    int total_channels = channels_A + channels_B + channels_C;

    return !dead &&
           energy >= div_threshold &&
           total_channels >= 2;
}


void Bacterium::record_switch_transition(const std::string& old_type,
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


static std::string choose_division_switch_type(const std::string& old_type) {
    if (old_type == "A") {
        return dist(gen) < 0.5 ? "B" : "C";
    }

    if (old_type == "B") {
        return dist(gen) < 0.5 ? "A" : "C";
    }

    return dist(gen) < 0.5 ? "A" : "B";
}


Bacterium* Bacterium::divide(int new_id, StepStats& stats) {
    if (dead) {
        return nullptr;
    }

    stats.divisions++;

    float r = std::clamp(
        static_cast<float>(norm(gen)),
        DIVISION_BIAS_MIN,
        DIVISION_BIAS_MAX
    );

    float child_energy = energy * (1.0f - r);
    energy *= r;

    auto split = [r](int count) -> std::pair<int, int> {
        int parent_count = static_cast<int>(std::round(count * r));
        int child_count = count - parent_count;

        parent_count = std::max(parent_count, 0);
        child_count = std::max(child_count, 0);

        return {parent_count, child_count};
    };

    auto [newA, child_A] = split(channels_A);
    auto [newB, child_B] = split(channels_B);
    auto [newC, child_C] = split(channels_C);

    channels_A = newA;
    channels_B = newB;
    channels_C = newC;

    std::string old_type = type;
    std::string child_type = type;

    // -------------------------------------------------------------------------
    // Model 1: true inherited phase variation.
    //
    // Only PHASE_MODEL == 1 can switch during division.
    // PHASE_MODEL == 0 and PHASE_MODEL == 2 do not use inherited switching.
    // -------------------------------------------------------------------------
    if (PHASE_MODEL == 1) {
        stats.switch_attempts++;

        if (dist(gen) < DIVISION_SWITCH_PROB) {
            child_type = choose_division_switch_type(old_type);

            if (child_type != old_type) {
                stats.switch_successes++;
                record_switch_transition(old_type, child_type, stats);
            } else {
                stats.switch_failures++;
            }
        } else {
            stats.switch_failures++;
        }
    }

    return new Bacterium(
        child_energy,
        new_id,
        child_A,
        child_B,
        child_C,
        child_type
    );
}