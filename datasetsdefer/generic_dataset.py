import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GenericImageExpertDataset(Dataset):
    def __init__(self, images, targets, expert_preds, transforms_fn, to_open=False):
        """

        Args:
            images (list): List of images
            targets (list): List of labels
            expert_preds (list): List of expert predictions
            transforms_fn (function): Function to apply to images
            to_open (bool): Whether to open images or not (RGB reader)
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_preds = np.array(expert_preds)
        self.transforms_fn = transforms_fn
        self.to_open = to_open

    def __getitem__(self, index):
        """Return the image, label, and expert prediction at this index."""
        label = self.targets[index]
        if self.transforms_fn is not None and self.to_open:
            image_paths = self.images[index]
            image = Image.open(image_paths).convert("RGB")
            image = self.transforms_fn(image)
        elif self.transforms_fn is not None:
            image = self.transforms_fn(self.images[index])
        else:
            image = self.images[index]
        expert_pred = self.expert_preds[index]
        return torch.FloatTensor(image), label, expert_pred

    def __len__(self):
        return len(self.targets)
