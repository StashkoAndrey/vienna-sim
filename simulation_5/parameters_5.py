# parameters_extracted.py
#
# CLEAN BASELINE PARAMETERS FOR simulation_5.
# Replace the old cluster-derived parameters_extracted.py with this file before
# running presentation experiments.

GRID_SIZE = 10
MAX_STEPS = 1000
REPLICATES = 10

REFILL_INTERVAL_A = 99
REFILL_INTERVAL_B = 99
REFILL_INTERVAL_C = 99
DELTA_T = 33

MEAL_TOTAL_A = 140
MEAL_TOTAL_B = 140
MEAL_TOTAL_C = 140
INITIAL_NUTRIENT = 140

DIFFUSION_RATE = 0.05

INITIAL_ENERGY = 5.0

# One-time cost paid only when a new channel is successfully constructed.
CHANNEL_COST = 0.3

DIVISION_THRESHOLD = 15.0

# Recurring survival cost paid by every living bacterium.
BASE_METABOLIC_COST = 0.1

# Recurring per-step cost per already existing channel.
MAINTENANCE_COST = 0.1

UPTAKE_PER_CHANNEL = 5.0
BUILD_PROB_SLOPE = 1.0
SUPPRESSION_K = 100
DEATH_PROB = 0.001

# Extra death probability when the channel-building/invertase machinery fires.
# Kept with original spelling because existing code imports INERTASE_PENALTY.
INERTASE_PENALTY = 0.1

DIVISION_SWITCH_PROB = 0.5

DIVISION_BIAS_MEAN = 0.62
DIVISION_BIAS_SD = 0.07
DIVISION_BIAS_MIN = 0.50
DIVISION_BIAS_MAX = 0.74

PHAGE_A_DIFFUSION_RATE = 0.05
PHAGE_A_ADSORPTION_RATE = 0.002
PHAGE_A_BURST_SIZE = 10
PHAGE_A_DECAY_RATE = 0.01
PHAGE_A_LATENT_PERIOD = 3
INITIAL_PHAGE_A_CONCENTRATION = 1

PHAGE_B_DIFFUSION_RATE = 0.05
PHAGE_B_ADSORPTION_RATE = 0.002
PHAGE_B_BURST_SIZE = 10
PHAGE_B_DECAY_RATE = 0.01
PHAGE_B_LATENT_PERIOD = 3
INITIAL_PHAGE_B_CONCENTRATION = 1

PHAGE_C_DIFFUSION_RATE = 0.05
PHAGE_C_ADSORPTION_RATE = 0.002
PHAGE_C_BURST_SIZE = 10
PHAGE_C_DECAY_RATE = 0.01
PHAGE_C_LATENT_PERIOD = 3
INITIAL_PHAGE_C_CONCENTRATION = 1

# 1.0 = pure A/B/C pulses.
# <1.0 = focal nutrient gets this fraction; the remainder is randomly split
# between the other two nutrients.
MAJOR_MEAL_FRACTION = 1.0

LOG_STEPS = False
