from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class DegradedTextDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, image_size=128):
        self.degraded_paths = sorted([
            os.path.join(degraded_dir, f)
            for f in os.listdir(degraded_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.clean_paths = sorted([
            os.path.join(clean_dir, f)
            for f in os.listdir(clean_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return min(len(self.degraded_paths), len(self.clean_paths))

    def __getitem__(self, idx):
        degraded = Image.open(self.degraded_paths[idx])
        clean = Image.open(self.clean_paths[idx])

        return self.transform(degraded), self.transform(clean)