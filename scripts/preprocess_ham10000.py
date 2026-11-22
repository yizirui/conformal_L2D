"""Prepare image and expert-metadata splits for the HAM10000 experiments."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm


LABELS = {"bkl": 0, "df": 1, "mel": 2, "nv": 3, "vasc": 4, "akiec": 5, "bcc": 6}
DIAGNOSIS_TYPES = {"histo": 0, "follow_up": 1, "consensus": 2, "confocal": 3}
GENDERS = {"male": 0, "female": 1, "unknown": 2}
IMAGE_SIZE = 224
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def read_metadata(directory):
    image_dir = directory / "HAM10000_images"
    filenames = {path.name for path in image_dir.iterdir()}
    with (directory / "HAM10000_metadata.csv").open(newline="") as source:
        rows = [
            row
            for row in csv.DictReader(source, skipinitialspace=True)
            if f"{row['image_id']}.jpg" in filenames and row["age"] != ""
        ]
    paths = [image_dir / f"{row['image_id']}.jpg" for row in rows]
    columns = {
        "labels": np.array([LABELS[row["dx"]] for row in rows]),
        "diagnosis_type": np.array([DIAGNOSIS_TYPES[row["dx_type"]] for row in rows]),
        "age": np.array([float(row["age"]) for row in rows]),
        "gender": np.array([GENDERS[row["sex"]] for row in rows]),
    }
    return paths, columns


def split_indices(labels, seed, legacy_split):
    # Both stratified draws use the same random stream as the original preprocessing.
    rng = np.random.RandomState(seed)
    pool, test = train_test_split(
        np.arange(len(labels)),
        test_size=0.15 if legacy_split else 0.20,
        stratify=labels,
        random_state=rng,
    )
    train, validation = train_test_split(
        pool, test_size=0.25, stratify=labels[pool], random_state=rng
    )
    return {"train": train, "validation": validation, "test": test}


def save_split(destination, name, indices, paths, columns):
    # Fill one split in output order, so the full image corpus never needs a second copy.
    images = torch.empty((len(indices), 3, IMAGE_SIZE, IMAGE_SIZE))
    for position, row in enumerate(tqdm(indices, desc=name)):
        with Image.open(paths[row]) as image:
            images[position] = PREPROCESS(image)
    metadata = {key: values[indices] for key, values in columns.items()}
    if name == "test":
        # The released test file stores metadata as tensors. Train and validation use NumPy.
        metadata = {key: torch.from_numpy(values) for key, values in metadata.items()}
    torch.save({"data": images, **metadata}, destination / f"{name}_data.pt")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare stratified HAM10000 train/validation/test splits (60/20/20)"
    )
    parser.add_argument(
        "directory", type=Path, help="Folder containing HAM10000_metadata.csv and HAM10000_images/"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy-split",
        action="store_true",
        help="Reproduce Multi_L2D's 15%% test split and hold out 25%% of the remainder for validation.",
    )
    args = parser.parse_args(argv)

    paths, columns = read_metadata(args.directory)
    splits = split_indices(columns["labels"], args.seed, args.legacy_split)
    # Finish reading every image before replacing any previously prepared file.
    with TemporaryDirectory(dir=args.directory) as staging:
        destination = Path(staging)
        for name, indices in splits.items():
            save_split(destination, name, indices, paths, columns)
        for name in splits:
            filename = f"{name}_data.pt"
            (destination / filename).replace(args.directory / filename)


if __name__ == "__main__":
    main()
