# bacteria_p.py

import numpy as np
import random
from parameters_p import (
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
    DIVISION_SWITCH_PROB,
    DIVISION_BIAS_MEAN,
    DIVISION_BIAS_SD,
    DIVISION_BIAS_MIN,
    DIVISION_BIAS_MAX
)

class Bacterium:
    def __init__(
        self,
        energy=INITIAL_ENERGY,
        cell_id=0,
        channels_A=0,
        channels_B=1,
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
        0) Random-death: with probability DEATH_PROB, die immediately.
        1) Uptake from BOTH A and B (if the cell has channels_A and/or channels_B).
           Add (uptake_A + uptake_B) to energy.
        2) Pay metabolic + channel costs ∝ (channels_A + channels_B).
        3) If energy > 0.2*DIVISION_THRESHOLD, attempt to build exactly one new channel 
           of self.type whenever local_{type} > 0 (build-prob uses build_channel_probability).
        """
        # 0) Random-death
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
         - AND total channels (A + B) ≥ 2 (so we never produce a zero-channel child).
        """
        total_ch = self.channels_A + self.channels_B
        return (not self.dead) and (self.energy >= self.division_threshold) and (total_ch >= 2)

    def divide(self, new_id):
        """
        Division logic with biased partition:
          1) Sample r ∼ Normal(DIVISION_BIAS_MEAN, DIVISION_BIAS_SD), then clamp to
             [DIVISION_BIAS_MIN, DIVISION_BIAS_MAX].  That fraction r is what the
             mother retains; 1−r goes to the child (for BOTH energy and channels).
          2) Mother_energy = self.energy * r,  Child_energy = self.energy * (1−r).
          3) For each channel-type (A and B): mother keeps round(ch_total · r), child
             gets the remainder.  This ensures the mother always keeps at least
             floor(r·ch_total), and the child gets ch_total - floor(r·ch_total).
          4) Channels are reassigned accordingly.
          5) Daughter’s type is chosen based on DIVISION_SWITCH_PROB.
        """
        if self.dead:
            return None

        # 1) Sample a biased split ratio r ∼ Normal(DIVISION_BIAS_MEAN, DIVISION_BIAS_SD),
        #    then clamp it so 0.50 ≤ r ≤ 0.74.
        r = np.random.normal(loc=DIVISION_BIAS_MEAN, scale=DIVISION_BIAS_SD)
        r = max(DIVISION_BIAS_MIN, min(DIVISION_BIAS_MAX, r))

        # 2) Partition energy according to r
        total_energy = self.energy
        parent_energy = total_energy * r
        child_energy  = total_energy * (1.0 - r)
        self.energy   = parent_energy

        # 3) Partition A-channels and B-channels deterministically (round + remainder)
        total_A = self.channels_A
        parent_A = int(round(total_A * r))
        child_A  = total_A - parent_A

        total_B = self.channels_B
        parent_B = int(round(total_B * r))
        child_B  = total_B - parent_B

        # 4) Re-assign the parent's channel pools to the “parent_” counts
        self.channels_A = parent_A
        self.channels_B = parent_B

        # 5) Decide the daughter's “type” with probability DIVISION_SWITCH_PROB, else same as parent
        if random.random() < DIVISION_SWITCH_PROB:
            new_type = 'B' if self.type == 'A' else 'A'
        else:
            new_type = self.type

        # 6) Construct the child with the “remainder” channels and child_energy
        child = Bacterium(
            energy=child_energy,
            cell_id=new_id,
            channels_A=child_A,
            channels_B=child_B,
            btype=new_type
        )
        return child
