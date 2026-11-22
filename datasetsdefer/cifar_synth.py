import random

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from .basedataset import BaseDataset
from .cifar_transforms import cifar_image_transform
from .generic_dataset import GenericImageExpertDataset


class CifarSynthExpert:
    """Know labels below k and guess uniformly among all classes otherwise."""

    def __init__(self, k, n_classes, random_seed=None):
        self.k = k
        self.n_classes = n_classes
        self.rng = random.Random(random_seed) if random_seed is not None else None

    def predict(self, labels):
        rng = getattr(self, "rng", None) or random
        return [label if label < self.k else rng.randrange(self.n_classes) for label in labels]


class CifarSynthDataset(BaseDataset):
    """This is the CifarK synthetic expert on top of Cifar-10 from Consistent Estimators for Learning to Defer (https://arxiv.org/abs/2006.01862)"""

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
        legacy_split=False,
    ):
        """
        expert_k: number of classes expert can predict
        use_data_aug: whether to use data augmentation (bool)
        test_split: fraction of all images reserved for testing, including the official test set
        val_split: fraction of all images reserved for calibration
        legacy_split: use the official test set and take val_split from the official training set
        batch_size: batch size for training
        """
        self.use_data_aug = use_data_aug
        self.data_dir = data_dir
        self.n_dataset = n_dataset
        self.expert_fn = CifarSynthExpert(expert_k, self.n_dataset).predict
        self.test_split = test_split
        self.val_split = val_split
        self.legacy_split = legacy_split or n_dataset != 10
        self.batch_size = batch_size
        if val_split != 0.1:
            self.dataset_name = f"Cifar10Synth{expert_k}_val{val_split}"
        else:
            self.dataset_name = f"Cifar10Synth{expert_k}"
        if not self.legacy_split:
            self.dataset_name += f"_paper_test{test_split}"
        self.generate_data()

    def generate_data(self):
        """Split the clean source images and sample an expert response for each row."""
        transform_train = cifar_image_transform(augment=self.use_data_aug)
        transform_test = cifar_image_transform()

        if self.n_dataset == 10:
            dataset = "cifar10"
        elif self.n_dataset == 100:
            dataset = "cifar100"

        train_dataset_all = datasets.__dict__[dataset.upper()](
            f"{self.data_dir}", train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            f"{self.data_dir}", train=False, transform=transform_test, download=True
        )
        if self.legacy_split:
            train_size = int((1 - self.val_split) * len(train_dataset_all))
            val_size = len(train_dataset_all) - train_size
            extra_test_size = 0
        else:
            # CIFAR-10-C contains corrupted copies of the official test images.
            # Keep all of them held out and draw any extra test rows from the training file.
            total_size = len(train_dataset_all) + len(test_dataset)
            val_size = int(self.val_split * total_size)
            extra_test_size = int(self.test_split * total_size) - len(test_dataset)
            train_size = len(train_dataset_all) - val_size - extra_test_size
            if extra_test_size < 0 or train_size < 1 or val_size < 1:
                raise ValueError("CIFAR needs nonempty training and calibration splits and "
                                 "enough test rows to hold the official test set.")

        train_subset, calibration_subset, extra_test_subset = torch.utils.data.random_split(
            train_dataset_all, [train_size, val_size, extra_test_size]
        )

        source_labels = np.asarray(train_dataset_all.targets)

        def annotated_split(subset, transform):
            images = train_dataset_all.data[subset.indices]
            labels = source_labels[subset.indices]
            return GenericImageExpertDataset(images, labels, self.expert_fn(labels), transform)

        dataset_train = annotated_split(train_subset, transform_train)
        dataset_val = annotated_split(calibration_subset, transform_test)
        test_images, test_labels = test_dataset.data, test_dataset.targets
        if extra_test_size:
            test_images = np.concatenate((test_images, train_dataset_all.data[extra_test_subset.indices]))
            test_labels = np.concatenate((test_labels, source_labels[extra_test_subset.indices]))
        dataset_test = GenericImageExpertDataset(test_images, test_labels, self.expert_fn(test_labels), transform_test)

        self.data_train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.data_val_loader = DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.data_test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
