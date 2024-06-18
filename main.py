import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data import DatasetSign
from data import DataSign2
from models import Generator, Discriminator
from training import train_dcgan, evaluate_model
from utils import visualize_metrics_seaborn, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import os

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 10
    batch_size = 32
    save_dir = 'images/batch_size_32/1/'

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    dataset1 = DatasetSign(root='dataset/archive (1)/CEDAR', name_dataset='CEDAR', transform=transform)

    dataset2 = DatasetSign(root='dataset/archive (1)/BHSig260-Hindi', name_dataset='BHSig260-Hindi', transform=transform)

    dataset3 = DatasetSign(root='dataset/archive (1)/BHSig260-Bengali', name_dataset='BHSig260-Bengali', transform=transform)

    datasets = [('CEDAR', dataset1), ('BHSig260-Hindi', dataset2), ('BHSig260-Bengali', dataset3)]
    print(len(dataset1), len(dataset2), len(dataset3))
    #
    generators = []
    discriminators = []
    metrics = []
    classification_reports = []
    #
    for dataset_name, dataset in datasets:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
        criterion = torch.nn.BCELoss()
        optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f'Training on {dataset_name}')
        generator, discriminator = train_dcgan(generator, discriminator, train_loader, num_epochs, device, criterion, optimizerD, optimizerG)
        labels, predictions, optimal_threshold, metric = evaluate_model(discriminator, test_loader, device, save_dir=save_dir, save_name=f"{dataset_name}_AUC_ROC.png")

        generators.append(generator)
        discriminators.append(discriminator)
        metrics.append((dataset_name, metric))
        classification_reports.append((dataset_name, labels, predictions, optimal_threshold))
        cm = confusion_matrix(labels, predictions)
        plot_confusion_path = f"{save_dir+dataset_name}_confusion.png"
        plot_confusion_matrix(cm, class_names=['True 0', 'True 1'], save_path=plot_confusion_path)

    # Train data custom

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = torch.nn.BCELoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    train_dataset = DataSign2(root='dataset/data', name_dataset='train', transform=transform)
    test_dataset = DataSign2(root='dataset/data', name_dataset='test', transform=transform)
    print(len(train_dataset), len(test_dataset))

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Training on Custom')
    generator, discriminator = train_dcgan(generator, discriminator, train_loader, num_epochs, device, criterion,
                                           optimizerD, optimizerG)
    labels, predictions, optimal_threshold, metric = evaluate_model(discriminator, test_loader, device, save_dir=save_dir, save_name="Custom_AUC_ROC.png")

    generators.append(generator)
    discriminators.append(discriminator)
    metrics.append(("Custom", metric))
    classification_reports.append(("Custom", labels, predictions, optimal_threshold))
    plot_confusion_path = save_dir+"Custom_confusion.png"
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, class_names=['True 0', 'True 1'], save_path=plot_confusion_path)

    for dataset_name, metric in metrics:
        print(f'Results for {dataset_name}:')
        plot_metrics_path = f"{save_dir+dataset_name}_metric.png"
        print(visualize_metrics_seaborn(metric, save_path=plot_metrics_path))
