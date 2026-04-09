import os
from src.data.downloader import DatasetDownloader
from src.data.loader import DataLoaderFactory
from src.models.evaluator import ModelEvaluator
from src.optimization.bayesian_optimizer import BayesianOptimizer
from src.utils.visualizer import OptimizationVisualizer

class OptimizationRunner:
    def __init__(self, train_path, test_path, total_trials=50):
        self.train_path = train_path
        self.test_path = test_path
        self.total_trials = total_trials
        self.downloader = DatasetDownloader()
        self.data_loader_factory = DataLoaderFactory(train_path, test_path)
        self.evaluator = ModelEvaluator()
        self.optimizer = BayesianOptimizer(total_trials=total_trials)

    def execute(self):
        self.downloader.download_and_extract()
        
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            print("Dataset not found.")
            return

        train_loader = self.data_loader_factory.create_train_loader()
        test_loader = self.data_loader_factory.create_test_loader()

        self.optimizer.execute_optimization(self.evaluator, train_loader, test_loader)
        
        best_accuracy, best_parameters = self.optimizer.extract_best_results()
        
        print("\nOptimization Finished.")
        print(f"Final Best Accuracy: {best_accuracy:.4f}")
        print(f"Final Best Parameters: {best_parameters}\n")

        OptimizationVisualizer.render_history_plot(self.optimizer.study)

if __name__ == "__main__":
    TRAINING_CSV_PATH = "datamunge/sign-language-mnist/sign_mnist_train.csv"
    TESTING_CSV_PATH = "datamunge/sign-language-mnist/sign_mnist_test.csv"
    
    optimization_instance = OptimizationRunner(TRAINING_CSV_PATH, TESTING_CSV_PATH, total_trials=20)
    optimization_instance.execute()