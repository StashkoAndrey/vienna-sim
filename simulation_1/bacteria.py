class Bacterium:
    def __init__(self, energy=3.0, uptake=1.0, channel_cost=0.2):
        self.energy = energy
        self.uptake = uptake
        self.channel_cost = channel_cost
        self.base_metabolic_cost = 0.1
        self.division_threshold = 5.0

    def consume(self, nutrient):
        uptake_amount = nutrient * self.uptake
        self.energy += uptake_amount
        # pay for channel maintenance and metabolic cost:
        self.energy -= uptake_amount * self.channel_cost
        self.energy -= self.base_metabolic_cost * self.uptake
        # channels grow proportionally to nutrient uptake
        self.uptake += 0.01 * uptake_amount

    def ready_to_divide(self):
        return self.energy >= self.division_threshold

    def divide(self):
        self.energy /= 2
        return Bacterium(
            energy=self.energy,
            uptake=self.uptake,
            channel_cost=self.channel_cost
        )
