import torch
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(discriminator, test_loader, device):
    all_labels = []
    all_predictions = []
    discriminator.eval()
    with torch.no_grad():
        for real_images, labels in test_loader:
            real_images = real_images.to(device)
            labels = labels.to(device).float()
            outputs = discriminator(real_images).view(-1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

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
    plt.show()

    all_predictions = [1 if x > optimal_threshold else 0 for x in all_predictions]
    classification_reports = classification_report(all_labels, all_predictions)
    print(classification_reports)
    classification_report_dict = classification_report(all_labels, all_predictions, output_dict=True)

    metrics = {
        "accuracy": classification_report_dict["accuracy"],
        "precision": classification_report_dict["0.0"]["precision"],
        "recall": classification_report_dict["0.0"]["recall"],
        "f1-score": classification_report_dict["0.0"]["f1-score"]
    }

    # visualize_metrics_seaborn(metrics)
    discriminator.train()

    return all_labels, all_predictions, optimal_threshold, metrics
