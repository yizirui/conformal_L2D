"""Construct and train the models used in the deferral experiments."""

import json
import math
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
import torch
import yaml
from torch import optim
from torch.utils.data import TensorDataset

from baselines.asym_lce_surrogate import AsymLceSurrogate
from baselines.one_v_all import OVASurrogate
from networks.cnn import NetSimple
from networks.linear_net import LinearNet

ROOT = Path(__file__).resolve().parents[1]
HAM_BACKBONE = Path("weights/ham10k_resnet34_100epochs_3407.pt")


@dataclass(frozen=True)
class TrainingSettings:
    optimizer: str
    epochs: int
    lr: float
    scheduler: Optional[str] = "CosineAnnealingLR"
    weight_decay: float = 0.0
    momentum: float = 0.0
    nesterov: bool = False

    def fit_options(self):
        optimizers = {"Adam": optim.Adam, "SGD": optim.SGD}
        schedulers = {None: None, "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR}
        if self.optimizer not in optimizers or self.scheduler not in schedulers:
            raise ValueError("Use optimizer Adam or SGD and scheduler CosineAnnealingLR or null.")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("Training epochs must be a positive integer.")
        for name in ("lr", "momentum", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Training {name} must be a finite non-negative number.")
        if type(self.nesterov) is not bool:
            raise ValueError("Training nesterov must be true or false.")
        if self.optimizer != "SGD" and (self.momentum or self.nesterov):
            raise ValueError("Only SGD accepts momentum and nesterov.")
        if self.nesterov and self.momentum == 0:
            raise ValueError("Nesterov SGD requires positive momentum.")

        kwargs = {"weight_decay": self.weight_decay}
        if self.optimizer == "SGD":
            kwargs.update(momentum=self.momentum, nesterov=self.nesterov)
        return {
            "optimizer": partial(optimizers[self.optimizer], **kwargs),
            "epochs": self.epochs,
            "lr": self.lr,
            "scheduler": schedulers[self.scheduler],
        }


DATASET_DEFAULTS = {
    "CIFAR10": TrainingSettings("Adam", 100, 1e-3),
    "HAM10000": TrainingSettings("SGD", 150, 1e-3),
    "HateSpeech": TrainingSettings("Adam", 50, 1e-2),
}


def _default_training_settings(dataset, surrogate):
    settings = DATASET_DEFAULTS[dataset]
    if surrogate == "ALCE":
        return replace(settings, weight_decay=5e-4)
    if dataset == "HAM10000":
        return replace(settings, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if dataset == "CIFAR10":
        return replace(settings, scheduler=None)
    return settings


def _training_settings(args):
    settings = _default_training_settings(args.dataset, args.surrogate)
    path = getattr(args, "training_config", None)
    if path is not None:
        with Path(path).open() as source:
            profiles = yaml.safe_load(source)
        if not isinstance(profiles, dict):
            raise ValueError(f"{path}: expected a mapping of dataset names to training settings.")
        unknown_datasets = profiles.keys() - DATASET_DEFAULTS.keys()
        if unknown_datasets:
            raise ValueError(f"{path}: unknown datasets: {sorted(unknown_datasets)}")
        profile = profiles.get(args.dataset, {})
        if not isinstance(profile, dict) or not isinstance(profile.get(args.surrogate, {}), dict):
            raise ValueError(f"{path}: expected a mapping for {args.dataset} and {args.surrogate}.")
        overrides = {k: v for k, v in profile.items() if k not in ("OvA", "ALCE")}
        overrides.update(profile.get(args.surrogate, {}))
        unknown = overrides.keys() - asdict(settings).keys()
        if unknown:
            raise ValueError(f"{path}: unknown training settings for {args.dataset}: {sorted(unknown)}")
        # PyYAML can read scientific notation such as 1e-3 as a string.
        for name in ("lr", "weight_decay", "momentum"):
            if name in overrides and type(overrides[name]) in (int, str):
                overrides[name] = float(overrides[name])
        settings = replace(settings, **overrides)
    return settings if args.epochs is None else replace(settings, epochs=args.epochs)


def _data_key(dataset, data_dir):
    """Detect a regenerated split even when the data directory and run seed are reused."""
    digest = sha256(str(data_dir.resolve()).encode())
    if isinstance(dataset, TensorDataset):  # Hate Speech embeddings
        arrays = dataset.tensors
    else:  # CIFAR and HAM GenericImageExpertDataset
        arrays = (dataset.images, dataset.targets, dataset.expert_preds)
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str((array.shape, array.dtype.str)).encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()[:20]


def _build_model(args, dataset, restore_weights):
    num_logits = dataset.n_dataset + 1
    if args.dataset == "CIFAR10":
        return NetSimple(num_logits, width1=50, width2=50, ff_units1=100, ff_units2=20)
    if args.dataset == "HateSpeech":
        return LinearNet(dataset.d, num_logits)

    from networks.modules import ClassifierRejector
    from networks.resnet224 import ResNet34

    weights = ROOT / HAM_BACKBONE if args.paper_ham_backbone else "imagenet"
    backbone = ResNet34(weights=None if restore_weights else weights)
    return ClassifierRejector(backbone, dataset.n_dataset, backbone.n_features, with_softmax=False)


def _cache_path(args, dataset, settings):
    cache_dir = args.checkpoints_dir / dataset.dataset_name
    train_key = _data_key(dataset.data_train_loader.dataset, args.data_dir)
    cache_dir = cache_dir / "full_train_final_epoch" / train_key
    if args.surrogate == "ALCE":
        cache_dir = cache_dir / "asm_eq6"
    defaults = _default_training_settings(args.dataset, args.surrogate)
    if settings != replace(defaults, epochs=settings.epochs):
        recipe = json.dumps(asdict(settings), sort_keys=True).encode()
        cache_dir = cache_dir / f"training_{sha256(recipe).hexdigest()[:20]}"
    elif args.surrogate == "ALCE" and args.dataset in ("CIFAR10", "HateSpeech"):
        # Earlier A-SM checkpoints at the same epoch count used SGD.
        cache_dir = cache_dir / "adam"
    if args.dataset == "HAM10000" and args.paper_ham_backbone:
        cache_dir = cache_dir / "paper_backbone"
    return cache_dir / (
        f"{args.model}_{args.surrogate}_ep{settings.epochs}_batch{args.batch_size}_seed{args.random_seed}.pt"
    )


def _save_checkpoint(path, surrogate):
    """Publish the complete surrogate weights in one rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save({"weights": surrogate.model.state_dict()}, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_or_train_surrogate(args, dataset, device):
    """Load explicit or cached weights, or fit the surrogate on the source training split."""
    if args.surrogate == "ALCE" and device.type != "cuda":
        raise ValueError("ALCE training requires CUDA in the upstream implementation.")
    if args.checkpoint:
        cache = None
    else:
        settings = _training_settings(args)
        options = settings.fit_options()
        cache = _cache_path(args, dataset, settings)
    cached = cache is not None and cache.is_file()
    model = _build_model(args, dataset, restore_weights=bool(args.checkpoint) or cached).to(device)
    if args.surrogate == "ALCE":
        surrogate = AsymLceSurrogate(1, 300, model, device)
    elif args.surrogate == "OvA":
        surrogate = OVASurrogate(1, 300, model, device)
    else:
        raise ValueError(f"Unknown conformal surrogate: {args.surrogate}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        return surrogate

    if cached:
        saved = torch.load(cache, map_location=device)
        model.load_state_dict(saved["weights"])
        return surrogate

    # D2 is reserved for conformal calibration.
    surrogate.fit(dataset.data_train_loader, None, None, verbose=False, **options)
    _save_checkpoint(cache, surrogate)
    return surrogate


# Earlier experiment scripts import this name.
fit_model = load_or_train_surrogate
