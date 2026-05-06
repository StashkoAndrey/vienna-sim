# parameters_5.py

# Central configuration for simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
# Grid and simulation structure
GRID_SIZE = 10
MAX_STEPS = 1000
REPLICATES = 10
# ─────────────────────────────────────────────────────────────────────────────
# Nutrient refill intervals and amounts for A vs. B vs. C
REFILL_INTERVAL_A = 99
REFILL_INTERVAL_B = 99
REFILL_INTERVAL_C = 99
DELTA_T = 33
MEAL_TOTAL_A = 140
MEAL_TOTAL_B = 140
MEAL_TOTAL_C = 140
INITIAL_NUTRIENT = 140
DIFFUSION_RATE = 0.05
# ─────────────────────────────────────────────────────────────────────────────
# Bacterium behavior
INITIAL_ENERGY = 5.0
CHANNEL_COST = 0.3
DIVISION_THRESHOLD = 15.0
BASE_METABOLIC_COST = 0.1
MAINTENANCE_COST = 0.1
UPTAKE_PER_CHANNEL = 5.0
BUILD_PROB_SLOPE = 1.0
SUPPRESSION_K = 100
DEATH_PROB = 0.001
INERTASE_PENALTY = 0.1
# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic switch at division
DIVISION_SWITCH_PROB = 0.5
# ─────────────────────────────────────────────────────────────────────────────
# Biased‐division parameters
DIVISION_BIAS_MEAN   = 0.62
DIVISION_BIAS_SD     = 0.07
DIVISION_BIAS_MIN    = 0.50
DIVISION_BIAS_MAX    = 0.74
# ─────────────────────────────────────────────────────────────────────────────
# Phage‐A (specific to A‐type bacteria)
PHAGE_A_DIFFUSION_RATE       = 0.05
PHAGE_A_ADSORPTION_RATE      = 0.002
PHAGE_A_BURST_SIZE           = 10
PHAGE_A_DECAY_RATE           = 0.01
PHAGE_A_LATENT_PERIOD        = 3
INITIAL_PHAGE_A_CONCENTRATION = 1
# ─────────────────────────────────────────────────────────────────────────────
# Phage‐B (specific to B‐type bacteria)
PHAGE_B_DIFFUSION_RATE       = 0.05
PHAGE_B_ADSORPTION_RATE      = 0.002
PHAGE_B_BURST_SIZE           = 10
PHAGE_B_DECAY_RATE           = 0.01
PHAGE_B_LATENT_PERIOD        = 3
INITIAL_PHAGE_B_CONCENTRATION = 1
# ─────────────────────────────────────────────────────────────────────────────
# Phage‐C (specific to C‐type bacteria)
PHAGE_C_DIFFUSION_RATE       = 0.05
PHAGE_C_ADSORPTION_RATE      = 0.002
PHAGE_C_BURST_SIZE           = 10
PHAGE_C_DECAY_RATE           = 0.01
PHAGE_C_LATENT_PERIOD        = 3
INITIAL_PHAGE_C_CONCENTRATION = 1
# ─────────────────────────── RANDOM MEAL SPLIT ───────────────────────────
# Fraction (0-1) of the meal allotted to the nutrient that is “on duty”
# during that refill.  The remainder is split randomly between the other two.
MAJOR_MEAL_FRACTION = 1.0        
LOG_STEPS = True  # Set to False to disable logging output
