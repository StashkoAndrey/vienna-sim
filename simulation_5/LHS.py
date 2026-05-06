from scipy.stats import qmc
import numpy as np
import pandas as pd

# Исходные переменные
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
CHANNEL_COST = 0.3
DIVISION_THRESHOLD = 15.0
BASE_METABOLIC_COST = 0.1
MAINTENANCE_COST = 0.1
UPTAKE_PER_CHANNEL = 5.0
BUILD_PROB_SLOPE = 1.0
SUPPRESSION_K = 100
DEATH_PROB = 0.001
DIVISION_SWITCH_PROB = 0.01
DIVISION_BIAS_MEAN   = 0.62
DIVISION_BIAS_SD     = 0.07
DIVISION_BIAS_MIN    = 0.50
DIVISION_BIAS_MAX    = 0.74
PHAGE_A_DIFFUSION_RATE       = 0.05
PHAGE_A_ADSORPTION_RATE      = 0.002
PHAGE_A_BURST_SIZE           = 10
PHAGE_A_DECAY_RATE           = 0.01
PHAGE_A_LATENT_PERIOD        = 3
INITIAL_PHAGE_A_CONCENTRATION = 1
PHAGE_B_DIFFUSION_RATE       = 0.05
PHAGE_B_ADSORPTION_RATE      = 0.002
PHAGE_B_BURST_SIZE           = 10
PHAGE_B_DECAY_RATE           = 0.01
PHAGE_B_LATENT_PERIOD        = 3
INITIAL_PHAGE_B_CONCENTRATION = 1
PHAGE_C_DIFFUSION_RATE       = 0.05
PHAGE_C_ADSORPTION_RATE      = 0.002
PHAGE_C_BURST_SIZE           = 10
PHAGE_C_DECAY_RATE           = 0.01
PHAGE_C_LATENT_PERIOD        = 3
INITIAL_PHAGE_C_CONCENTRATION = 1
MAJOR_MEAL_FRACTION = 1        

# Словарь со значениями
parameters = {
    var: (1 if value <= 1.0 else 1000)
    for var, value in locals().items()
    if var.isupper()  # только переменные в верхнем регистре
}


def lhs(n_samples, parameters):
    # Создание Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=len(parameters))
    # Генерация выборки
    sample = sampler.random(n=n_samples)
    # Масштабирование выборки
    scaled_sample = qmc.scale(sample, np.zeros(len(parameters)), np.array(list(parameters.values())))
    df = pd.DataFrame(scaled_sample, columns=list(parameters.keys()))
    # Добавляем колонку с номерами строк (от 1 до 100) как первый столбец
    df.insert(0, "number of simulation", range(1, 10001))
    # Сохраняем в CSV с 10 знаками после запятой
    df.to_csv("parameters_2.csv", index=False, float_format="%.10f")

lhs(10000, parameters)
