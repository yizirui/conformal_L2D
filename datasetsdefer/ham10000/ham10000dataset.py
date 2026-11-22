"""Read Multi_L2D's image, diagnosis, and metadata tensors for the HAM expert."""
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ham10000_expert(Dataset):
    def __init__(self, img_data, labels, diagnosis_type, age, gender, data_aug=False):
        super().__init__()
        self.data = img_data
        self.labels = labels
        self.diagnosis_type = diagnosis_type
        self.age = age
        self.gender = gender
        self.transform = None
        if data_aug:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    @classmethod
    def read(cls, path="./Data/", data_aug=False):
        """Read the clean splits used to annotate images with the metadata expert."""
        splits = []
        for name in ("train_data.pt", "validation_data.pt", "test_data.pt"):
            saved = torch.load(Path(path) / name)
            splits.append(cls(saved["data"], saved["labels"], saved["diagnosis_type"], saved["age"], saved["gender"]))
        return tuple(splits)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        metadata = torch.tensor([self.diagnosis_type[index], self.age[index], self.gender[index]])
        return image, self.labels[index], metadata

    def __len__(self):
        return self.data.shape[0]
