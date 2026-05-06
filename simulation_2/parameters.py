# Central configuration for simulation parameters

# Grid and simulation structure
GRID_SIZE = 10
MAX_STEPS = 1000
REPLICATES = 10
MEAL_INTERVAL =100
MEAL_TOTAL = 150
INITIAL_NUTRIENT = 150
DIFFUSION_RATE = 0.05

# Bacterium behavior
INITIAL_ENERGY = 5.0
CHANNEL_COST = 0.3
DIVISION_THRESHOLD = 15.0
BASE_METABOLIC_COST = 0.1
MAINTENANCE_COST = 0.1
UPTAKE_PER_CHANNEL = 5.0
BUILD_PROB_SLOPE = 1.0         # tune to adjust how aggressively channels build
SUPPRESSION_K    = 100          # higher → existing channels suppress builds more slowly
BUILD_CHANNEL_BEFORE_COST = True  # set False to test post‐cost building
