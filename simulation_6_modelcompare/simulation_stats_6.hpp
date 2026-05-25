#ifndef SIMULATION_STATS_6_HPP
#define SIMULATION_STATS_6_HPP

// Per-timestep event counters + derived ecological/phase-variation metrics.
// Event counters are filled during SimulationGrid::step(). Derived metrics are
// filled after diffusion/refill by compute_step_metrics().
struct StepStats {
    // Cell events
    int divisions = 0;
    int switch_attempts = 0;
    int switch_successes = 0;
    int switch_failures = 0;  // currently 0 because sim6 has no failure mechanism yet

    int switch_A_to_B = 0;
    int switch_A_to_C = 0;
    int switch_B_to_A = 0;
    int switch_B_to_C = 0;
    int switch_C_to_A = 0;
    int switch_C_to_B = 0;

    // Phage events
    int infection_attempts_A = 0;
    int infection_attempts_B = 0;
    int infection_attempts_C = 0;

    int infection_successes_A = 0;
    int infection_successes_B = 0;
    int infection_successes_C = 0;

    int lysis_events_A = 0;
    int lysis_events_B = 0;
    int lysis_events_C = 0;

    // Population / phenotype state
    int total_population = 0;
    int pop_A = 0;
    int pop_B = 0;
    int pop_C = 0;

    int pop_major_A = 0;
    int pop_major_B = 0;
    int pop_major_C = 0;

    int variant_richness = 0;               // number of majority variants present: 0..3
    float shannon_entropy_majority = 0.0f;  // entropy over majority A/B/C counts
    float simpson_diversity_majority = 0.0f;// 1 - sum(p_i^2)
    float minor_variant_fraction = 0.0f;    // 1 - max majority frequency

    // Channel state
    float mean_channels_total = 0.0f;
    float mean_channels_A = 0.0f;
    float mean_channels_B = 0.0f;
    float mean_channels_C = 0.0f;

    // System totals
    float total_nutrient_A = 0.0f;
    float total_nutrient_B = 0.0f;
    float total_nutrient_C = 0.0f;

    float total_phage_A = 0.0f;
    float total_phage_B = 0.0f;
    float total_phage_C = 0.0f;

    float total_energy = 0.0f;
};

#endif // SIMULATION_STATS_6_HPP
