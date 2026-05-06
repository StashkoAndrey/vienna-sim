# parameters_p.py

# Central configuration for simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
# Grid and simulation structure
GRID_SIZE = 10
MAX_STEPS = 1000
REPLICATES = 10
# ─────────────────────────────────────────────────────────────────────────────
# Nutrient refill intervals and amounts for A vs. B
REFILL_INTERVAL_A = 100
REFILL_INTERVAL_B = 100
DELTA_T = 50
MEAL_TOTAL_A = 140
MEAL_TOTAL_B = 140
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
# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic switch at division
DIVISION_SWITCH_PROB = 0.5
# ─────────────────────────────────────────────────────────────────────────────
#Biased‐division parameters
DIVISION_BIAS_MEAN   = 0.62    # mean fraction mother keeps
DIVISION_BIAS_SD     = 0.07    # std dev for the normal draw
DIVISION_BIAS_MIN    = 0.50    # clamp lower bound
DIVISION_BIAS_MAX    = 0.74    # clamp upper bound
# ─────────────────────────────────────────────────────────────────────────────
# Phage‐A (specific to A‐type bacteria)
# ─────────────────────────────────────────────────────────────────────────────
PHAGE_A_DIFFUSION_RATE       = 0.05    # diffusion coefficient for phage‐A field
PHAGE_A_ADSORPTION_RATE      = 0.002   # prob. per‐unit‐phage that an A‐cell is infected
PHAGE_A_BURST_SIZE           = 10      # new phage‐A particles released upon lysis of an A‐cell
PHAGE_A_DECAY_RATE           = 0.01    # fraction of phage‐A that decays each timestep
PHAGE_A_LATENT_PERIOD        = 3       # # of timesteps between infection & lysis for A‐cells
INITIAL_PHAGE_A_CONCENTRATION = 1      # starting phage‐A concentration everywhere
# ─────────────────────────────────────────────────────────────────────────────
# Phage‐B (specific to B‐type bacteria)
# ─────────────────────────────────────────────────────────────────────────────
PHAGE_B_DIFFUSION_RATE       = 0.05    # diffusion coefficient for phage‐B field
PHAGE_B_ADSORPTION_RATE      = 0.002   # prob. per‐unit‐phage that a B‐cell is infected
PHAGE_B_BURST_SIZE           = 10      # new phage‐B particles released upon lysis of a B‐cell
PHAGE_B_DECAY_RATE           = 0.01    # fraction of phage‐B that decays each timestep
PHAGE_B_LATENT_PERIOD        = 3       # # of timesteps between infection & lysis for B‐cells
INITIAL_PHAGE_B_CONCENTRATION = 1      # starting phage‐B concentration everywhere