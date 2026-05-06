# bacteria_5.py
import random
import numpy as np

class Bacterium:
    def __init__(self, energy, cell_id, channels_A, channels_B, channels_C, btype, params):
        self.energy       = energy
        self.channels_A   = channels_A
        self.channels_B   = channels_B
        self.channels_C   = channels_C
        self.type         = btype
        self.id           = cell_id
        self.dead         = False
        self.params       = params  # Сохраняем словарь параметров

    def _build_prob(self, local_nutrient, total_channels):
        slope = self.params["BUILD_PROB_SLOPE"]
        suppression_k = self.params["SUPPRESSION_K"]
        nutr_drive = slope * local_nutrient
        suppression = total_channels / (total_channels + suppression_k)
        return max(0.0, min(1.0, nutr_drive * (1.0 - suppression)))

    def consume(self, local_A, local_B, local_C):
        p = self.params
        if random.random() < p["DEATH_PROB"]:
            self.dead = True
            return

        upA = min(local_A, self.channels_A * p["UPTAKE_PER_CHANNEL"])
        upB = min(local_B, self.channels_B * p["UPTAKE_PER_CHANNEL"])
        upC = min(local_C, self.channels_C * p["UPTAKE_PER_CHANNEL"])
        self.energy += (upA + upB + upC)

        tot_ch = self.channels_A + self.channels_B + self.channels_C
        self.energy -= (
            p["BASE_METABOLIC_COST"] +
            p["MAINTENANCE_COST"] * tot_ch +
            p["CHANNEL_COST"] * tot_ch
        )

        if self.energy > 0.2 * p["DIVISION_THRESHOLD"]:
            if self.type == 'A' and local_A > 0 and random.random() < self._build_prob(local_A, tot_ch) * min(local_A / p["INITIAL_NUTRIENT"], 1.0):
                self.channels_A += 1
            elif self.type == 'B' and local_B > 0 and random.random() < self._build_prob(local_B, tot_ch) * min(local_B / p["INITIAL_NUTRIENT"], 1.0):
                self.channels_B += 1
            elif self.type == 'C' and local_C > 0 and random.random() < self._build_prob(local_C, tot_ch) * min(local_C / p["INITIAL_NUTRIENT"], 1.0):
                self.channels_C += 1

        self.energy     = max(self.energy, 0.0)
        self.channels_A = max(self.channels_A, 0)
        self.channels_B = max(self.channels_B, 0)
        self.channels_C = max(self.channels_C, 0)

    def ready_to_divide(self):
        tot_ch = self.channels_A + self.channels_B + self.channels_C
        return not self.dead and self.energy >= self.params["DIVISION_THRESHOLD"] and tot_ch >= 2

    def divide(self, new_id):
        p = self.params
        r = np.clip(np.random.normal(p["DIVISION_BIAS_MEAN"], p["DIVISION_BIAS_SD"]), p["DIVISION_BIAS_MIN"], p["DIVISION_BIAS_MAX"])
        child_energy = self.energy * (1.0 - r)
        self.energy *= r

        def split(count):
            parent = int(round(count * r))
            return parent, count - parent

        self.channels_A, child_A = split(self.channels_A)
        self.channels_B, child_B = split(self.channels_B)
        self.channels_C, child_C = split(self.channels_C)

        if random.random() < p["DIVISION_SWITCH_PROB"]:
            new_type = random.choice({'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}[self.type])
        else:
            new_type = self.type

        return Bacterium(
            energy=child_energy, cell_id=new_id,
            channels_A=child_A, channels_B=child_B, channels_C=child_C,
            btype=new_type, params=p
        )
    def major_type(self):
        tA, tB, tC = self.channels_A, self.channels_B, self.channels_C
        if tA >= tB and tA >= tC:
            return 'A'
        elif tB >= tA and tB >= tC:
            return 'B'
        return 'C'
