import numpy as np
import random

from parameters import (
    INITIAL_ENERGY,
    CHANNEL_COST,
    DIVISION_THRESHOLD,
    BASE_METABOLIC_COST,
    MAINTENANCE_COST,
    UPTAKE_PER_CHANNEL,
    SUPPRESSION_K,
    BUILD_PROB_SLOPE,
    BUILD_CHANNEL_BEFORE_COST,
)


class Bacterium:
    def __init__(self, energy=INITIAL_ENERGY, cell_id=0, channels=1):
        self.energy = energy
        self.channels = channels
        self.id = cell_id

    def build_channel_probability(self, uptake_amount: float) -> float:
        """
        Probability ∝ log(1 + uptake) with diminishing returns from existing channels.
        Never reaches exactly zero.
        """
        drive = BUILD_PROB_SLOPE * np.log1p(uptake_amount)
        suppression = self.channels / (self.channels + SUPPRESSION_K)
        return float(max(0.0, min(1.0, drive * (1 - suppression))))

    def consume(self, uptake_amount: float, local_nutrient: float):
        """
        1) Add uptake to energy
        2) Optionally build channel (before costs)
        3) Subtract metabolic & channel maintenance costs
        4) Optionally build channel (after costs)
        5) Clamp energy & channels to non-negative
        """
        # 1) Uptake
        self.energy += uptake_amount

        # 2) Build channel before costs?
        if BUILD_CHANNEL_BEFORE_COST and uptake_amount > 0:
            if random.random() < self.build_channel_probability(uptake_amount):
                self.channels += 1

        # 3) Pay costs
        total_metabolic = BASE_METABOLIC_COST + MAINTENANCE_COST * self.channels
        total_channel   = CHANNEL_COST       * self.channels
        self.energy    -= (total_metabolic + total_channel)

        # 4) Build channel after costs?
        if not BUILD_CHANNEL_BEFORE_COST and uptake_amount > 0:
            if random.random() < self.build_channel_probability(uptake_amount):
                self.channels += 1

        # 5) Clamp
        self.channels = max(1, self.channels)
        self.energy   = max(0.0, self.energy)

    def ready_to_divide(self) -> bool:
        """True if energy meets the threshold defined in parameters."""
        return self.energy >= DIVISION_THRESHOLD

    def divide(self, new_id: int) -> "Bacterium":
        """
        Split energy & channels randomly between parent and offspring.
        Child goes into an empty neighbor cell.
        """
        energy_ratio  = np.random.uniform(0.3, 0.7)
        channel_ratio = np.random.uniform(0.3, 0.7)

        child_energy   = self.energy * energy_ratio
        child_channels = max(1, int(self.channels * channel_ratio))

        # Parent keeps the rest
        self.energy   *= (1 - energy_ratio)
        self.channels -= child_channels

        return Bacterium(
            energy=child_energy,
            cell_id=new_id,
            channels=child_channels
        )
