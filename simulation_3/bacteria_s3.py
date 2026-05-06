# bacteria_s3.py
import numpy as np
import random
from parameters_s3 import (
    INITIAL_ENERGY,
    CHANNEL_COST,
    DIVISION_THRESHOLD,
    BASE_METABOLIC_COST,
    MAINTENANCE_COST,
    UPTAKE_PER_CHANNEL,
    SUPPRESSION_K,
    BUILD_PROB_SLOPE,
    INITIAL_NUTRIENT,
    DEATH_PROB,
    DIVISION_SWITCH_PROB,      # <— import the new switch‐prob
)

class Bacterium:
    def __init__(
        self,
        energy=INITIAL_ENERGY,
        cell_id=0,
        channels_A=1,
        channels_B=0,
        btype='A',
    ):
        """
        Each Bacterium:
         - tracks channels_A ≥ 0 and channels_B ≥ 0.
         - has a type 'A' or 'B', which determines which nutrient it can build channels for.
        """
        self.energy     = energy
        self.channels_A = channels_A
        self.channels_B = channels_B
        self.type       = btype    # 'A' or 'B'
        self.channel_cost      = CHANNEL_COST
        self.maintenance_cost  = MAINTENANCE_COST
        self.base_metabolic_cost = BASE_METABOLIC_COST
        self.division_threshold  = DIVISION_THRESHOLD
        self.id = cell_id
        self.dead = False

    def build_channel_probability(self, local_nutrient, total_channels):
        """
        Probability of building one new channel of this cell’s own type,
        based on local nutrient concentration and suppression by total_channels.
        """
        nutrient_drive = BUILD_PROB_SLOPE * local_nutrient
        suppression    = total_channels / (total_channels + SUPPRESSION_K)
        prob = nutrient_drive * (1 - suppression)
        return float(max(0.0, min(1.0, prob)))

    def consume(self, local_A, local_B):
        """
        0) Random‐death: with probability DEATH_PROB, die immediately.
        1) Uptake from BOTH A and B (if the cell has channels_A and/or channels_B).
           Add (uptake_A + uptake_B) to energy.
        2) Pay metabolic + channel costs ∝ (channels_A + channels_B).
        3) If energy > 0.2*DIVISION_THRESHOLD, attempt to build exactly one new channel 
           of self.type whenever local_{type} > 0 (build‐prob uses build_channel_probability).
        """
        # 0) Random‐death
        if random.random() < DEATH_PROB:
            self.dead = True
            return

        # 1) Uptake A & B
        uptake_A = min(local_A, self.channels_A * UPTAKE_PER_CHANNEL)
        uptake_B = min(local_B, self.channels_B * UPTAKE_PER_CHANNEL)
        self.energy += (uptake_A + uptake_B)

        # 2) Pay costs (metabolic + channel)
        total_ch = self.channels_A + self.channels_B
        total_metabolic    = self.base_metabolic_cost + self.maintenance_cost * total_ch
        total_channel_cost = self.channel_cost * total_ch
        self.energy -= (total_metabolic + total_channel_cost)

        # 3) Build one new channel of self.type if energy threshold and local nutrient > 0
        energy_threshold = DIVISION_THRESHOLD * 0.2
        if self.energy > energy_threshold:
            total_ch = self.channels_A + self.channels_B
            if self.type == 'A' and local_A > 0:
                prob_A = self.build_channel_probability(local_A, total_ch)
                nut_factor_A = min(local_A / INITIAL_NUTRIENT, 1.0)
                if random.random() < (nut_factor_A * prob_A):
                    self.channels_A += 1

            elif self.type == 'B' and local_B > 0:
                prob_B = self.build_channel_probability(local_B, total_ch)
                nut_factor_B = min(local_B / INITIAL_NUTRIENT, 1.0)
                if random.random() < (nut_factor_B * prob_B):
                    self.channels_B += 1

        # Clamp to avoid negatives
        self.channels_A = max(0, self.channels_A)
        self.channels_B = max(0, self.channels_B)
        self.energy     = max(0.0, self.energy)

    def ready_to_divide(self):
        """
        True if:
         - not dead,
         - energy ≥ DIVISION_THRESHOLD,
         - AND total channels (A + B) ≥ 2 (so we never produce a zero‐channel child).
        """
        total_ch = self.channels_A + self.channels_B
        return (not self.dead) and (self.energy >= self.division_threshold) and (total_ch >= 2)

    def divide(self, new_id):
        """
        Division logic with probabilistic type switch:
        1) Randomly split energy (30–70% to child).
        2) Independently toss a fair coin for each existing A‐channel → parent_A vs. child_A.
        3) Independently toss a fair coin for each existing B‐channel → parent_B vs. child_B.
        4) Re‐assign parent’s channels_A := parent_A and channels_B := parent_B.
        5) Daughter inherits channels_A = child_A, channels_B = child_B.
        6) With probability DIVISION_SWITCH_PROB, daughter.type := opposite(parent.type),
           otherwise daughter.type := parent.type (i.e., no flip).
        7) Return the new child instance.
        """
        if self.dead:
            return None

        # 1) Split energy
        energy_ratio = np.random.uniform(0.3, 0.7)
        child_energy = self.energy * energy_ratio
        self.energy   *= (1 - energy_ratio)

        # 2) Partition each A‐channel
        parent_A = 0
        child_A  = 0
        for _ in range(self.channels_A):
            if random.random() < 0.5:
                parent_A += 1
            else:
                child_A += 1

        # 3) Partition each B‐channel
        parent_B = 0
        child_B  = 0
        for _ in range(self.channels_B):
            if random.random() < 0.5:
                parent_B += 1
            else:
                child_B += 1

        # 4) Re‐assign parent’s channel pools
        self.channels_A = parent_A
        self.channels_B = parent_B

        # 5) Construct the child’s base channel counts
        #    (child_A, child_B) exactly as partitioned above.
        # 6) Decide daughter’s type with probability DIVISION_SWITCH_PROB
        if random.random() < DIVISION_SWITCH_PROB:
            # Flip to opposite of parent
            new_type = 'B' if self.type == 'A' else 'A'
        else:
            # Keep same type as parent
            new_type = self.type

        child = Bacterium(
            energy=child_energy,
            cell_id=new_id,
            channels_A=child_A,
            channels_B=child_B,
            btype=new_type,
        )
        return child
