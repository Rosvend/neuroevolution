import matplotlib.pyplot as plt

class TrainingVisualizer:
    @staticmethod
    def plot_fitness_evolution(history):
        plt.figure(figsize=(10, 6))
        plt.plot(history, marker='o')
        plt.title('Genetic Algorithm Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Best Accuracy (Fitness)')
        plt.grid(True)
        plt.show()
