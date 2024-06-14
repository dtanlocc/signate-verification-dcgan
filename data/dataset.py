import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class DatasetSign(Dataset):
    def __init__(self,root: str, name_dataset: str, transform=None):
        self.root = root
        self.transform = transform

        self.dir = os.path.join(self.root, name_dataset)
        # print(self.dir)
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        # print(self.dir)
        self.folder = glob(os.path.join(self.dir, "*/"))
        # print(self.folder)
        for index in self.folder:
            index_path = os.path.join(index)
            if os.path.isdir(index_path):
                for img_name in os.listdir(index_path):
                    img_path = os.path.join(index_path, img_name)
                    if 'original' in img_name or 'G' in img_name:
                        self.data.append(img_path)
                        self.labels.append(1)  # 1 cho chữ ký thật
                    elif 'forgeries' in img_name or 'F' in img_name:
                        self.data.append(img_path)
                        self.labels.append(0)  # 0 cho chữ ký giả

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # Chuyển sang grayscale
        if self.transform:
            image = self.transform(image)
        return image, label
