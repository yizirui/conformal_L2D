"""Connect each experiment to its dataset and expert."""

import os
import random


def load_dataset(args, severity, corruption, expert_k, device):
    if args.dataset == "CIFAR10":
        if severity == 0:
            from datasetsdefer.cifar_synth import CifarSynthDataset

            return CifarSynthDataset(
                expert_k, use_data_aug=False, data_dir=str(args.data_dir),
                batch_size=args.batch_size, val_split=args.val_split,
                legacy_split=args.legacy_cifar_split,
            )

        if severity == 6:
            from datasetsdefer.svhn_synth import SVHNSynthDataset

            return SVHNSynthDataset(
                expert_k, data_dir=args.data_dir, batch_size=args.shifted_batch_size,
                val_split=args.val_split, random_seed=args.random_seed,
            )

        from datasetsdefer.cifar10c_synth import Cifar10CorruptionDataset

        required = [f"CIFAR-10-C/{corruption}.npy", "CIFAR-10-C/labels.npy"]
        for name in required:
            path = args.data_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. See docs/data.md for CIFAR-10-C.")
        return Cifar10CorruptionDataset(
            expert_k, use_data_aug=False, data_dir=str(args.data_dir) + os.sep,
            batch_size=args.shifted_batch_size, corruption=corruption, severity=severity,
            random_seed=args.random_seed, val_split=args.val_split,
        )

    if args.dataset == "HAM10000":
        from datasetsdefer.ham10k import HAM10kSynthDataset

        return HAM10kSynthDataset(
            use_data_aug=False, data_dir=str(args.data_dir / "HAM10000") + os.sep,
            device=device, severity=severity, random_seed=args.random_seed, batch_size=args.batch_size,
        )

    from datasetsdefer.hatespeech import HateSpeech

    (args.data_dir / "hatespeech").mkdir(parents=True, exist_ok=True)
    # Keep each tweet in the clean split. Only the noise uses the severity-specific seed.
    split_seed = random.Random(args.random_seed).randrange(10000)
    return HateSpeech(
        str(args.data_dir), embed_texts=True, include_demographics=False,
        expert_type=args.annotator, device=device, severity=severity,
        batch_size=args.batch_size, split_seed=split_seed, random_seed=args.random_seed,
    )
