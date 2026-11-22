"""SVHN evaluation data for the CIFAR-10 semantic shift in Figure 2."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN

from .cifar_synth import CifarSynthExpert
from .cifar_transforms import cifar_image_transform
from .generic_dataset import GenericImageExpertDataset


class SVHNSynthDataset:
    """Calibrate on a subset of SVHN train and evaluate on the full official test split."""

    def __init__(self, expert_k, *, data_dir, batch_size, val_split, random_seed):
        self.dataset_name = f"SVHNSynth{expert_k}"
        self.n_dataset = 10
        transform = cifar_image_transform()

        def load_split(split):
            data = SVHN(root=str(data_dir), split=split, download=True)
            # Torchvision stores CHW bytes and already maps the SVHN digit 10 to 0.
            images = data.data.transpose(0, 2, 3, 1)
            labels = data.labels
            # Separate streams keep each image's answer independent of loader order and val_split.
            expert = CifarSynthExpert(expert_k, self.n_dataset, f"svhn/{random_seed}/{split}")
            answers = np.asarray(expert.predict(labels), dtype=np.int64)
            if split == "train":
                count = int(val_split * len(labels))
                if not 0 < count < len(labels):
                    raise ValueError("SVHN val_split must select a nonempty proper subset of train.")
                generator = torch.Generator().manual_seed(random_seed)
                rows = torch.randperm(len(labels), generator=generator)[-count:].numpy()
                images, labels, answers = images[rows], labels[rows], answers[rows]
            dataset = GenericImageExpertDataset(images, labels, answers, transform)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.data_val_loader = load_split("train")
        self.data_test_loader = load_split("test")
