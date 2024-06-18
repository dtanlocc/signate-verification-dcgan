import matplotlib.pyplot as plt
# import os
def plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label='Optimal threshold = %0.2f' % optimal_threshold)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.show()
    print(save_path)
    plt.savefig(save_path)
    plt.close()
