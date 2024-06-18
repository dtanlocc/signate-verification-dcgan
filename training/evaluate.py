import torch
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from project.utils import plot_roc_curve


def evaluate_model(discriminator, test_loader, device: str, save_dir: str, save_name: str):
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

    plot_file_path = save_dir+save_name
    plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, plot_file_path)

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
