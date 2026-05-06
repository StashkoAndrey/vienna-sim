import numpy as np
import random
from parameters import INITIAL_ENERGY, CHANNEL_COST, DIVISION_THRESHOLD, BASE_METABOLIC_COST, MAINTENANCE_COST, UPTAKE_PER_CHANNEL, SUPPRESSION_K, BUILD_PROB_SLOPE, INITIAL_NUTRIENT

class Bacterium:
    def __init__(self, energy=INITIAL_ENERGY, cell_id=0, channels=1):
        self.energy = energy
        self.channels = channels
        self.channel_cost = CHANNEL_COST
        self.maintenance_cost = MAINTENANCE_COST
        self.base_metabolic_cost = BASE_METABOLIC_COST
        self.division_threshold = DIVISION_THRESHOLD
        self.id = cell_id

    def build_channel_probability(self, local_nutrient):
        """
        Compute probability of building a channel based on local nutrient
        and channel suppression.
        """
        nutrient_drive = BUILD_PROB_SLOPE * local_nutrient
        suppression = self.channels / (self.channels + SUPPRESSION_K)
        prob = nutrient_drive * (1 - suppression)
        return float(max(0.0, min(1.0, prob)))

    def consume(self, uptake_amount, local_nutrient):
        """
        Uptake nutrient, pay metabolic & channel costs, then attempt
        to build a new channel stochastically.
        """
        # 1) Gain from uptake
        self.energy += uptake_amount
        # 2) Pay metabolic costs
        total_metabolic = self.base_metabolic_cost + MAINTENANCE_COST * self.channels
        total_channel   = self.channel_cost * self.channels
        self.energy -= (total_metabolic + total_channel)

        # 3) Attempt channel build based on local nutrient AND energy reserve
        #    Higher local nutrient increases chance; low reserves block it
        energy_threshold = DIVISION_THRESHOLD * 0.2
        if uptake_amount > 0 and self.energy > energy_threshold:
            # Combine nutrient availability and suppression into one probability
            # Scale nutrient to [0,1] by dividing by INITIAL_NUTRIENT
            nut_factor = min(local_nutrient / INITIAL_NUTRIENT, 1.0)
            base_prob = self.build_channel_probability(local_nutrient)
            prob = nut_factor * base_prob
            if random.random() < prob:
                self.channels += 1
                
        # clamp
        self.channels = max(0, self.channels)
        self.energy = max(0.0, self.energy)

    def ready_to_divide(self):
        return self.energy >= self.division_threshold

    def divide(self, new_id):
        energy_ratio  = np.random.uniform(0.3, 0.7)
        channel_ratio = np.random.uniform(0.3, 0.7)
        child_energy   = self.energy * energy_ratio
        child_channels = max(1, int(self.channels * channel_ratio))
        # parent retains the remainder
        self.energy   *= (1 - energy_ratio)
        self.channels -= child_channels
        return Bacterium(energy=child_energy, cell_id=new_id, channels=child_channels)
