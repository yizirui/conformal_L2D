"""Prepare HAM10000 image, label and metadata-expert prediction triples."""
import os
from errno import EEXIST, ENOTEMPTY
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .basedataset import BaseDataset
from .generic_dataset import GenericImageExpertDataset
from .ham10000.ham10000dataset import ham10000_expert
from .ham10000.models.experts import EXPERT_WEIGHTS, MetadataExpert
from .shift_transforms import GaussianBlurTransform, HideTransform


def _float_image(image):
    # HideTransform masks in place. Give each access its own copy of the image.
    return torch.tensor(image, dtype=torch.float32)


def _pad_image(image):
    tensor = torch.as_tensor(image, dtype=torch.float32)
    return F.pad(tensor.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)


def _pil_to_float(image):
    # Keep the original pixel scale, with channels first for the crop and flip.
    return transforms.functional.pil_to_tensor(image).float()


def _image_transform(severity, augment=False):
    hide = HideTransform(level=severity)
    blur = GaussianBlurTransform(severity=severity)
    if augment:
        return transforms.Compose([
            _pad_image, transforms.ToPILImage(), _pil_to_float,
            transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), blur, hide,
        ])
    return transforms.Compose([_float_image, hide, blur])


def _file_version(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=32)
def _file_digest(path, version):
    # A sweep hashes each multi-GB file once. An edit or replacement changes the key.
    digest = sha256()
    with path.open("rb") as source:
        if _file_version(os.fstat(source.fileno())) != version:
            raise RuntimeError(f"HAM10000 input changed while opening {path}.")
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        if _file_version(os.fstat(source.fileno())) != version:
            raise RuntimeError(f"HAM10000 input changed while reading {path}.")
    return digest.digest()


class HAM10kSynthDataset(BaseDataset):
    """Use the prepared train/validation/test splits and the Multi_L2D expert."""

    def __init__(
        self, use_data_aug, data_dir, device, alpha=0.95, severity=0,
        test_split=0.15, val_split=0.25, batch_size=1000, n_dataset=7,
        transforms=None, random_seed=42,
    ):
        self.data_dir = data_dir
        self.random_seed = random_seed
        root = Path(data_dir).resolve()
        source_paths = [root / name for name in ("train_data.pt", "validation_data.pt", "test_data.pt")]
        self._source_paths = (*source_paths, EXPERT_WEIGHTS)
        self._versions = self._input_versions()
        digest = sha256()
        for path, version in zip(self._source_paths, self._versions):
            digest.update(_file_digest(path, version))
        self.source_key = digest.hexdigest()
        # Earlier annotation caches discarded the final partial training batch.
        self.cache_dir = root / "cache" / "full_splits" / f"seed_{random_seed}_batch_{batch_size}" / self.source_key
        self.use_data_aug = use_data_aug
        self.n_dataset = n_dataset
        self.device = device
        self.expert_fn = MetadataExpert(device, EXPERT_WEIGHTS).MLPMixer
        self.batch_size = batch_size
        self.dataset_name = f"HAM10kSynth_MLP_Mixer_{severity}"
        self.generate_data(severity)

    def _input_versions(self):
        return tuple(_file_version(path.stat()) for path in self._source_paths)

    def _check_inputs(self):
        if self._input_versions() != self._versions:
            raise RuntimeError("HAM10000 inputs changed while preparing the dataset. Reload the splits.")

    def _loader(self, dataset, *, shuffle=False, generator=None):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True, generator=generator,
        )

    @torch.no_grad()
    def _annotate_split(self, dataset, *, training):
        count = len(dataset)
        if not count:
            raise ValueError("Cannot annotate an empty HAM10000 split.")
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = self._loader(dataset, shuffle=training, generator=generator)

        targets = []
        predictions = []
        offset = 0
        for images, labels, metadata in loader:
            expert_labels = self.expert_fn(
                images.to(self.device), labels.to(self.device), metadata.float().to(self.device),
            )
            batch = images.cpu().numpy()
            if offset == 0:
                # The image split is large. Fill its final buffer directly in loader order.
                split_images = np.empty((count, *batch.shape[1:]), dtype=batch.dtype)
            stop = offset + len(batch)
            split_images[offset:stop] = batch
            offset = stop
            targets.append(labels.cpu())
            predictions.append(expert_labels.cpu())

        return GenericImageExpertDataset(
            split_images, torch.cat(targets).numpy(), torch.cat(predictions).numpy(), None,
        )

    def generate_data(self, severity=0):
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        splits = ("train", "val", "test")
        eval_transform = _image_transform(severity)
        train_transform = _image_transform(severity, augment=True) if self.use_data_aug else eval_transform
        split_transforms = (train_transform, eval_transform, eval_transform)

        if self.cache_dir.is_dir():
            datasets = [torch.load(self.cache_dir / f"{split}.pt") for split in splits]
        else:
            source_splits = dict(zip(
                splits, ham10000_expert.read(self.data_dir),
            ))
            datasets = []
            # Publish all three clean splits together. Concurrent readers can only see complete sets.
            with TemporaryDirectory(dir=self.cache_dir.parent) as work_dir:
                prepared = Path(work_dir) / "annotated"
                prepared.mkdir()
                for split in splits:
                    dataset = self._annotate_split(source_splits.pop(split), training=(split == "train"))
                    torch.save(dataset, prepared / f"{split}.pt", pickle_protocol=5, pickle_module=dill)
                    datasets.append(dataset)
                self._check_inputs()
                try:
                    prepared.rename(self.cache_dir)
                except OSError as exc:
                    if exc.errno not in (EEXIST, ENOTEMPTY):
                        raise
                    # A concurrent writer published the same inputs and annotation seed first.
                    datasets.clear()
                    datasets = [torch.load(self.cache_dir / f"{split}.pt") for split in splits]

        self._check_inputs()
        for dataset, transform in zip(datasets, split_transforms):
            dataset.transforms_fn = transform
        train, val, test = datasets
        self.data_train_loader = self._loader(train, shuffle=True)
        self.data_val_loader = self._loader(val)
        self.data_test_loader = self._loader(test)
