import seaborn as sns
import matplotlib.pyplot as plt

def visualize_metrics_seaborn(metrics):
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics_names, y=metrics_values, palette='viridis')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Evaluation Metrics')
    plt.ylim([0, 1])

    # Hiển thị giá trị trên các cột
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

    plt.show()
