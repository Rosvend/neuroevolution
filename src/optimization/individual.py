import random

class Individual:
    def __init__(self, chromosome, fitness=0.0):
        self.chromosome = chromosome
        self.fitness = fitness

    @classmethod
    def generate_random(cls, max_layers=5, min_neurons=16, max_neurons=512):
        num_layers = random.randint(1, max_layers)
        chromosome = [random.randint(min_neurons, max_neurons) for _ in range(num_layers)]
        chromosome += [0] * (max_layers - num_layers)
        return cls(chromosome)
