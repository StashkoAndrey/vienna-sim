import random
import numpy as np

# Corrected simulation_5 bacterium logic.
# Intended model:
# - CHANNEL_COST is paid only once when a new channel is constructed.
# - MAINTENANCE_COST is paid every timestep for existing channels.
# - INERTASE_PENALTY blocks/fails division-time switching.
#   Therefore INERTASE_PENALTY = 1.0 means no switch can establish,
#   matching the behavior of DIVISION_SWITCH_PROB = 0.0.

from parameters_5 import (
    INITIAL_ENERGY, CHANNEL_COST, DIVISION_THRESHOLD,
    BASE_METABOLIC_COST, MAINTENANCE_COST,
    UPTAKE_PER_CHANNEL, SUPPRESSION_K, BUILD_PROB_SLOPE,
    INITIAL_NUTRIENT, DEATH_PROB, DIVISION_SWITCH_PROB,
    DIVISION_BIAS_MEAN, DIVISION_BIAS_SD,
    DIVISION_BIAS_MIN, DIVISION_BIAS_MAX,
    INERTASE_PENALTY
)


class Bacterium:
    def __init__(
        self,
        energy=INITIAL_ENERGY,
        cell_id=0,
        channels_A=0,
        channels_B=0,
        channels_C=0,
        btype='A',
        invertase_penalty=INERTASE_PENALTY,
        switch_prob=DIVISION_SWITCH_PROB,
    ):
        self.energy = energy
        self.channels_A = channels_A
        self.channels_B = channels_B
        self.channels_C = channels_C
        self.type = btype
        self.id = cell_id
        self.dead = False

        self.invertase_penalty = invertase_penalty
        self.switch_prob = switch_prob

        # Cost parameters.
        self.channel_cost = CHANNEL_COST
        self.maintenance_cost = MAINTENANCE_COST
        self.base_metabolic_cost = BASE_METABOLIC_COST
        self.div_threshold = DIVISION_THRESHOLD

    @staticmethod
    def _build_prob(local_nutrient, total_channels):
        """
        Nutrient-driven, self-limited probability of attempting construction.
        """
        nutr_drive = BUILD_PROB_SLOPE * local_nutrient
        suppression = total_channels / (total_channels + SUPPRESSION_K)
        return max(0.0, min(1.0, nutr_drive * (1.0 - suppression)))

    def _try_build_channel(self, channel_type, local_nutrient, total_channels):
        """
        Try to build one channel of a given type.

        CHANNEL_COST is paid only if a channel is actually built.
        INERTASE_PENALTY is NOT applied here, because in this corrected model
        invertase penalty belongs to division-time phenotype switching.
        """
        if local_nutrient <= 0:
            return

        prob = self._build_prob(local_nutrient, total_channels)
        prob *= min(local_nutrient / INITIAL_NUTRIENT, 1.0)

        if random.random() >= prob:
            return

        if self.energy < self.channel_cost:
            return

        if channel_type == 'A':
            self.channels_A += 1
        elif channel_type == 'B':
            self.channels_B += 1
        elif channel_type == 'C':
            self.channels_C += 1
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

        self.energy -= self.channel_cost

    def consume(self, local_A, local_B, local_C):
        """
        One bacterial metabolism step.

        Corrected logic:
        1. Gain energy from nutrients through existing channels.
        2. Pay recurring base metabolism and maintenance costs.
        3. Possibly build one new channel of the current type.
           CHANNEL_COST is paid only here.
        4. Apply baseline random death only.

        INERTASE_PENALTY is intentionally not used here.
        It is applied in divide(), where phenotype switching is attempted.
        """
        total_ch_before = self.channels_A + self.channels_B + self.channels_C

        # 1) Uptake through existing channels.
        upA = min(local_A, self.channels_A * UPTAKE_PER_CHANNEL)
        upB = min(local_B, self.channels_B * UPTAKE_PER_CHANNEL)
        upC = min(local_C, self.channels_C * UPTAKE_PER_CHANNEL)
        self.energy += (upA + upB + upC)

        # 2) Recurring costs.
        recurring_cost = (
            self.base_metabolic_cost
            + self.maintenance_cost * total_ch_before
        )
        self.energy -= recurring_cost
        self.energy = max(self.energy, 0.0)

        # 3) One-time channel construction cost, only on successful construction.
        if self.energy > 0.2 * self.div_threshold:
            total_ch_current = self.channels_A + self.channels_B + self.channels_C

            if self.type == 'A':
                self._try_build_channel('A', local_A, total_ch_current)
            elif self.type == 'B':
                self._try_build_channel('B', local_B, total_ch_current)
            elif self.type == 'C':
                self._try_build_channel('C', local_C, total_ch_current)

        # 4) Baseline random death.
        death_chance = max(0.0, min(1.0, DEATH_PROB))
        if random.random() < death_chance:
            self.dead = True
            return

        self.energy = max(self.energy, 0.0)
        self.channels_A = max(self.channels_A, 0)
        self.channels_B = max(self.channels_B, 0)
        self.channels_C = max(self.channels_C, 0)

    def ready_to_divide(self):
        total_ch = self.channels_A + self.channels_B + self.channels_C
        return (not self.dead) and self.energy >= self.div_threshold and total_ch >= 2

    def _daughter_type_after_switch_logic(self):
        """
        Division-time phenotype switching.

        DIVISION_SWITCH_PROB controls whether a switch attempt occurs.
        INERTASE_PENALTY controls whether the invertase/switching machinery fails.

        Therefore:
            effective_switch_probability
            = DIVISION_SWITCH_PROB * (1 - INERTASE_PENALTY)

        If INERTASE_PENALTY = 1.0, every switch attempt fails,
        so no switch establishes and behavior matches DIVISION_SWITCH_PROB = 0.0.
        """
        if random.random() >= DIVISION_SWITCH_PROB:
            return self.type

        # Switch attempted, but invertase penalty may block it.
        penalty = max(0.0, min(1.0, INERTASE_PENALTY))
        if random.random() < penalty:
            return self.type

        other = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}[self.type]
        return random.choice(other)

    def divide(self, new_id):
        if self.dead:
            return None

        r = np.clip(
            np.random.normal(DIVISION_BIAS_MEAN, DIVISION_BIAS_SD),
            DIVISION_BIAS_MIN,
            DIVISION_BIAS_MAX,
        )

        def split(count):
            parent = int(round(count * r))
            return parent, count - parent

        child_energy = self.energy * (1.0 - r)
        self.energy *= r

        self.channels_A, child_A = split(self.channels_A)
        self.channels_B, child_B = split(self.channels_B)
        self.channels_C, child_C = split(self.channels_C)

        new_type = self._daughter_type_after_switch_logic()

        return Bacterium(
            energy=child_energy,
            cell_id=new_id,
            channels_A=child_A,
            channels_B=child_B,
            channels_C=child_C,
            btype=new_type,
            invertase_penalty=INERTASE_PENALTY,
            switch_prob=DIVISION_SWITCH_PROB,
        )
