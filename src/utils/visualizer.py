import matplotlib.pyplot as plt

class OptimizationVisualizer:
    @staticmethod
    def render_history_plot(study):
        history_dataframe = study.trials_dataframe()
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_dataframe['number'], history_dataframe['value'], marker='o', linestyle='-', color='b')
        plt.title('Bayesian Optimization History')
        plt.xlabel('Trial Number')
        plt.ylabel('Accuracy Value')
        plt.grid(True)
        plt.show()