#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

// Grid and simulation structure
constexpr int GRID_SIZE = 10;
constexpr int MAX_STEPS = 200;
constexpr int REPLICATES = 20;

// Nutrient refill intervals and amounts for A vs. B vs. C
constexpr int REFILL_INTERVAL_A = 99;
constexpr int REFILL_INTERVAL_B = 99;
constexpr int REFILL_INTERVAL_C = 99;
constexpr int DELTA_T = 33;
constexpr float MEAL_TOTAL_A = 140.0f;
constexpr float MEAL_TOTAL_B = 140.0f;
constexpr float MEAL_TOTAL_C = 140.0f;
constexpr float INITIAL_NUTRIENT = 140.0f;
constexpr float DIFFUSION_RATE = 0.05f;

// Bacterium behavior
constexpr float INITIAL_ENERGY = 5.0f;
constexpr float CHANNEL_COST = 0.3f;
constexpr float DIVISION_THRESHOLD = 15.0f;
constexpr float BASE_METABOLIC_COST = 0.1f;
constexpr float MAINTENANCE_COST = 0.1f;
constexpr float UPTAKE_PER_CHANNEL = 5.0f;
constexpr float BUILD_PROB_SLOPE = 1.0f;
constexpr float SUPPRESSION_K = 100.0f;
constexpr float DEATH_PROB = 0.001f;

// Division switching
constexpr float DIVISION_SWITCH_PROB = 0.0;

// Biased‐division parameters
constexpr float DIVISION_BIAS_MEAN = 0.62f;
constexpr float DIVISION_BIAS_SD   = 0.07f;
constexpr float DIVISION_BIAS_MIN  = 0.50f;
constexpr float DIVISION_BIAS_MAX  = 0.74f;

// Phage‐A
constexpr float PHAGE_A_DIFFUSION_RATE  = 0.05f;
constexpr float PHAGE_A_ADSORPTION_RATE = 0.002f;
constexpr int   PHAGE_A_BURST_SIZE      = 10;
constexpr float PHAGE_A_DECAY_RATE      = 0.01f;
constexpr int   PHAGE_A_LATENT_PERIOD   = 3;
constexpr float INITIAL_PHAGE_A_CONCENTRATION = 1.0f;

// Phage‐B
constexpr float PHAGE_B_DIFFUSION_RATE  = 0.05f;
constexpr float PHAGE_B_ADSORPTION_RATE = 0.002f;
constexpr int   PHAGE_B_BURST_SIZE      = 10;
constexpr float PHAGE_B_DECAY_RATE      = 0.01f;
constexpr int   PHAGE_B_LATENT_PERIOD   = 3;
constexpr float INITIAL_PHAGE_B_CONCENTRATION = 1.0f;

// Phage‐C
constexpr float PHAGE_C_DIFFUSION_RATE  = 0.05f;
constexpr float PHAGE_C_ADSORPTION_RATE = 0.002f;
constexpr int   PHAGE_C_BURST_SIZE      = 10;
constexpr float PHAGE_C_DECAY_RATE      = 0.01f;
constexpr int   PHAGE_C_LATENT_PERIOD   = 3;
constexpr float INITIAL_PHAGE_C_CONCENTRATION = 1.0f;

// Random meal allocation
constexpr float MAJOR_MEAL_FRACTION = 1.0f;

// Logging
constexpr bool LOG_STEPS = false;

#endif // PARAMETERS_HPP
