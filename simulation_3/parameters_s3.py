# parameters_s3.py

# Central configuration for simulation parameters

# Grid and simulation structure
GRID_SIZE = 20
MAX_STEPS = 1000
REPLICATES = 10

# Nutrient refill intervals and amounts for A vs. B
REFILL_INTERVAL_A = 100
REFILL_INTERVAL_B = 100

MEAL_TOTAL_A = 140
MEAL_TOTAL_B = 140

INITIAL_NUTRIENT = 140
DIFFUSION_RATE = 0.05

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

# Probabilistic switch at division
DIVISION_SWITCH_PROB = 0.5

# -----------------------------------------------------------------------------
# Parameter‐Sweep Bounds (hardcoded here)
# -----------------------------------------------------------------------------
SWEEP_FREQ_MIN = 1        # minimum meal frequency
SWEEP_FREQ_MAX = 500      # maximum meal frequency
SWEEP_FREQ_STEP = 10      # step size for sampling frequencies

SWEEP_AMT_MIN = 1         # minimum meal amount
SWEEP_AMT_MAX = 100       # maximum meal amount
SWEEP_AMT_STEP = 2       # step size for sampling meal amounts

SWEEP_REPLICATES = 10     # number of stochastic replicates per (freq, amt) pair
SWEEP_MAX_STEPS = 1000    # run each simulation for up to 1000 steps when sweeping
