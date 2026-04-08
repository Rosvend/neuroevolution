import random
from src.optimization.individual import Individual

class GeneticOptimizer:
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [Individual.generate_random() for _ in range(population_size)]

    def select_parent(self, tournament_size=3):
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda individual: individual.fitness)

    def crossover(self, parent_1, parent_2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(0, len(parent_1.chromosome) - 1)
            child_1_chromosome = parent_1.chromosome[:crossover_point] + parent_2.chromosome[crossover_point:]
            child_2_chromosome = parent_2.chromosome[:crossover_point] + parent_1.chromosome[crossover_point:]
            return Individual(child_1_chromosome), Individual(child_2_chromosome)
        return Individual(parent_1.chromosome[:]), Individual(parent_2.chromosome[:])

    def mutate(self, individual, min_neurons=16, max_neurons=512):
        new_chromosome = individual.chromosome[:]
        for index in range(len(new_chromosome)):
            if random.random() < self.mutation_rate:
                new_chromosome[index] = random.randint(min_neurons, max_neurons)
        return Individual(new_chromosome)

    def run_generation(self, evaluator, train_loader, val_loader):
        for individual in self.population:
            if individual.fitness == 0.0:
                individual.fitness = evaluator.evaluate(individual.chromosome, train_loader, val_loader)

        new_population = []
        while len(new_population) < self.population_size:
            parent_1 = self.select_parent()
            parent_2 = self.select_parent()
            child_1, child_2 = self.crossover(parent_1, parent_2)
            new_population.append(self.mutate(child_1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child_2))

        self.population = new_population

    def get_best_individual(self):
        return max(self.population, key=lambda individual: individual.fitness)
