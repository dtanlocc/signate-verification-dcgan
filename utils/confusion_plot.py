import matplotlib.pyplot as plt
import seaborn as sns
# import os

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.show()
    print(save_path)
    plt.savefig(save_path)
    plt.close()