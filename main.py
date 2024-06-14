import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data import DatasetSign
from models import Generator, Discriminator
from training import train_dcgan, evaluate_model
from utils import visualize_metrics_seaborn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
beta1 = 0.5
num_epochs = 10
batch_size = 128

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset1 = DatasetSign(root='C:/Users/TanLoc/Desktop/signature_verification/project/dataset/archive (1)/CEDAR', name_dataset='CEDAR', transform=transform)
dataset2 = DatasetSign(root='C:/Users/TanLoc/Desktop/signature_verification/project/dataset/archive (1)/BHSig260-Hindi', name_dataset='BHSig260-Hindi', transform=transform)
dataset3 = DatasetSign(root='C:/Users/TanLoc/Desktop/signature_verification/project/dataset/archive (1)/BHSig260-Bengali', name_dataset='BHSig260-Bengali', transform=transform)
datasets = [('CEDAR', dataset1), ('BHSig260-Hindi', dataset2), ('BHSig260-Bengali', dataset3)]
print(len(dataset1),len(dataset2),len(dataset3))

generators = []
discriminators = []
metrics = []
classification_reports = []

for dataset_name, dataset in datasets:
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = torch.nn.BCELoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Training on {dataset_name}')
    generator, discriminator = train_dcgan(generator, discriminator, train_loader, beta1, num_epochs, lr, device, criterion, optimizerD, optimizerG)
    labels, predictions, optimal_threshold, metric = evaluate_model(discriminator, test_loader, device)

    generators.append(generator)
    discriminators.append(discriminator)
    metrics.append((dataset_name, metric))
    classification_reports.append((dataset_name, labels, predictions, optimal_threshold))

for dataset_name, metric in metrics:
    print(f'Results for {dataset_name}:')
    print(visualize_metrics_seaborn(metric))
