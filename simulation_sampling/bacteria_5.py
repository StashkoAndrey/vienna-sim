# Updated bacteria_5.py
import random
import numpy as np

class Bacterium:
    def __init__(self, energy, cell_id, channels_A, channels_B, channels_C,
                 btype, invertase_penalty, switch_prob, param_dict):
        self.energy = energy
        self.channels_A = channels_A
        self.channels_B = channels_B
        self.channels_C = channels_C
        self.type = btype
        self.id = cell_id
        self.dead = False
        self.invertase_worked = False
        self.invertase_penalty = invertase_penalty
        self.switch_prob = switch_prob
        self.param_dict = param_dict

        self.channel_cost = param_dict['CHANNEL_COST']
        self.maintenance_cost = param_dict['MAINTENANCE_COST']
        self.base_metabolic_cost = param_dict['BASE_METABOLIC_COST']
        self.div_threshold = param_dict['DIVISION_THRESHOLD']

    def _build_prob(self, local_nutrient, total_channels):
        slope = self.param_dict['BUILD_PROB_SLOPE']
        suppress_k = self.param_dict['SUPPRESSION_K']
        nutr_drive = slope * local_nutrient
        suppression = total_channels / (total_channels + suppress_k)
        return max(0.0, min(1.0, nutr_drive * (1.0 - suppression)))

    def consume(self, local_A, local_B, local_C):
        self.invertase_worked = False
        tot_ch = self.channels_A + self.channels_B + self.channels_C
        ini_nutr = self.param_dict['INITIAL_NUTRIENT']

        if self.energy > 0.2 * self.div_threshold:
            if self.type == 'A' and local_A > 0:
                prob = self._build_prob(local_A, tot_ch) * min(local_A / ini_nutr, 1.0)
                if random.random() < prob:
                    self.channels_A += 1
                    self.invertase_worked = True
            elif self.type == 'B' and local_B > 0:
                prob = self._build_prob(local_B, tot_ch) * min(local_B / ini_nutr, 1.0)
                if random.random() < prob:
                    self.channels_B += 1
                    self.invertase_worked = True
            elif self.type == 'C' and local_C > 0:
                prob = self._build_prob(local_C, tot_ch) * min(local_C / ini_nutr, 1.0)
                if random.random() < prob:
                    self.channels_C += 1
                    self.invertase_worked = True

        death_chance = self.param_dict['DEATH_PROB']
        if self.invertase_worked:
            death_chance += self.invertase_penalty
        if random.random() < death_chance:
            self.dead = True
            return

        uptake = self.param_dict['UPTAKE_PER_CHANNEL']
        upA = min(local_A, self.channels_A * uptake)
        upB = min(local_B, self.channels_B * uptake)
        upC = min(local_C, self.channels_C * uptake)
        self.energy += (upA + upB + upC)

        self.energy -= (
            self.base_metabolic_cost +
            self.maintenance_cost * tot_ch +
            self.channel_cost * tot_ch
        )

        self.energy = max(self.energy, 0.0)
        self.channels_A = max(self.channels_A, 0)
        self.channels_B = max(self.channels_B, 0)
        self.channels_C = max(self.channels_C, 0)

    def ready_to_divide(self):
        tot_ch = self.channels_A + self.channels_B + self.channels_C
        return (not self.dead) and self.energy >= self.div_threshold and tot_ch >= 2

    def divide(self, new_id):
        if self.dead:
            return None

        r = np.clip(
            np.random.normal(
                self.param_dict['DIVISION_BIAS_MEAN'],
                self.param_dict['DIVISION_BIAS_SD']
            ),
            self.param_dict['DIVISION_BIAS_MIN'],
            self.param_dict['DIVISION_BIAS_MAX']
        )

        def split(count):
            parent = int(round(count * r))
            return parent, count - parent

        child_energy = self.energy * (1.0 - r)
        self.energy *= r

        self.channels_A, child_A = split(self.channels_A)
        self.channels_B, child_B = split(self.channels_B)
        self.channels_C, child_C = split(self.channels_C)

        if random.random() < self.switch_prob:
            other = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}[self.type]
            new_type = random.choice(other)
        else:
            new_type = self.type

        return Bacterium(
            energy=child_energy, cell_id=new_id,
            channels_A=child_A, channels_B=child_B, channels_C=child_C,
            btype=new_type,
            invertase_penalty=self.invertase_penalty,
            switch_prob=self.switch_prob,
            param_dict=self.param_dict
        )

    def major_type(self):
        tA, tB, tC = self.channels_A, self.channels_B, self.channels_C
        if tA >= tB and tA >= tC:
            return 'A'
        elif tB >= tA and tB >= tC:
            return 'B'
        return 'C'
