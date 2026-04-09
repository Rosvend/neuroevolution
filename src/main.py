import os
from src.data.downloader import DatasetDownloader
from src.data.loader import DataLoaderFactory
from src.models.evaluator import ModelEvaluator
from src.optimization.genetic_algorithm import GeneticOptimizer
from src.utils.visualizer import TrainingVisualizer

class ExperimentRunner:
    def __init__(self, train_path, test_path, generations=10, population_size=10):
        self.train_path = train_path
        self.test_path = test_path
        self.generations = generations
        self.population_size = population_size
        self.downloader = DatasetDownloader()
        self.data_loader_factory = DataLoaderFactory(train_path, test_path)
        self.evaluator = ModelEvaluator()
        self.optimizer = GeneticOptimizer(population_size=population_size)
        self.fitness_history = []

    def run(self):
        self.downloader.download_and_extract()
        
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            print("Dataset not found. Please ensure sign_mnist_train.csv and sign_mnist_test.csv exist.")
            return

        train_loader = self.data_loader_factory.create_train_loader()
        test_loader = self.data_loader_factory.create_test_loader()

        for generation in range(self.generations):
            self.optimizer.run_generation(self.evaluator, train_loader, test_loader)
            best_individual = self.optimizer.get_best_individual()
            self.fitness_history.append(best_individual.fitness)
            print(f"Generation {generation+1}: Best Fitness = {best_individual.fitness:.4f}")

        TrainingVisualizer.plot_fitness_evolution(self.fitness_history)

if __name__ == "__main__":
    TRAIN_CSV = "datamunge/sign-language-mnist/sign_mnist_train.csv"
    TEST_CSV = "datamunge/sign-language-mnist/sign_mnist_test.csv"
    
    runner = ExperimentRunner(TRAIN_CSV, TEST_CSV)
    runner.run()
