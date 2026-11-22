from pathlib import Path

import numpy as np
import torch

from .basedataset import BaseDataset
from .cifar_synth import CifarSynthExpert
from .cifar_transforms import cifar_image_transform
from .generic_dataset import GenericImageExpertDataset


class Cifar10CorruptionDataset(BaseDataset):
    """CIFAR-10-C with simulated expert responses for each original test image."""

    def __init__(
        self,
        expert_k,
        use_data_aug,
        data_dir,
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        n_dataset=10,
        transforms=None,
        corruption="brightness",
        severity=1,
        random_seed=42,
    ):
        """Reserve val_split and test_split fractions of one corruption/severity block."""
        self.data_dir = Path(data_dir) / "CIFAR-10-C"
        self.use_data_aug = use_data_aug
        self.n_dataset = n_dataset
        self.expert_k = expert_k
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_per_severity = 10000
        self.dataset_name = f"Cifar10C_Synth{expert_k}_{corruption}_severity{severity}"
        self.corruption_types = [
            'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
            'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
            'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
            'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
            'zoom_blur'
        ]
        self.random_seed = random_seed
        self.generate_data(corruption, severity)

    def generate_data(self, corruption, severity):
        if corruption not in self.corruption_types:
            raise ValueError(f"Corruption type {corruption} not in CIFAR-10C dataset")

        start = self.num_per_severity * (severity - 1)
        stop = self.num_per_severity * severity
        cifar10c_data = np.load(self.data_dir / f"{corruption}.npy")[start:stop]
        cifar10c_labels = np.load(self.data_dir / "labels.npy")[start:stop].astype(np.int64)

        # Every corruption block has the same original CIFAR test-image order.
        # Draw once per original image before assigning calibration and test rows.
        expert = CifarSynthExpert(self.expert_k, self.n_dataset, self.random_seed)
        human_predictions = np.asarray(expert.predict(cifar10c_labels), dtype=np.int64)

        transform_train = cifar_image_transform(augment=self.use_data_aug)
        transform_test = cifar_image_transform()

        total_size = len(cifar10c_data)
        test_size = int(self.test_split * total_size)
        val_size = int(self.val_split * total_size)
        train_size = total_size - test_size - val_size

        order = torch.randperm(total_size, generator=torch.Generator().manual_seed(self.random_seed)).numpy()
        train, val, test = np.split(order, [train_size, train_size + val_size])

        def image_dataset(rows, transform):
            return GenericImageExpertDataset(
                cifar10c_data[rows], cifar10c_labels[rows], human_predictions[rows], transform,
            )

        data_train = image_dataset(train, transform_train)
        data_val = image_dataset(val, transform_test)
        data_test = image_dataset(test, transform_test)

        self.data_train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


# Historic experiment scripts use this name for the full CIFAR-10-C dataset.
Cifar10CHSynthExpert = Cifar10CorruptionDataset
