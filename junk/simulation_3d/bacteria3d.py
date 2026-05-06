# --- bacteria3d.py ---
class Bacterium3D:
    def __init__(self, energy=0.0, uptake_multiplier=1.0):
        self.energy = energy
        self.uptake_multiplier = uptake_multiplier
        self.division_threshold = 10.0
        self.channel_cost = 0.2

    def consume(self, nutrient):
        uptake = nutrient * self.uptake_multiplier
        self.energy += uptake
        self.energy -= self.channel_cost * uptake
        self.uptake_multiplier += 0.01 * uptake

    def ready_to_divide(self):
        return self.energy >= self.division_threshold

    def divide(self):
        self.energy /= 2
        return Bacterium3D(self.energy, self.uptake_multiplier)