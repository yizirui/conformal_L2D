"""Run a paper experiment from its YAML configuration."""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import yaml

from conformal import DEFERRAL_METHOD

ROOT = Path(__file__).resolve().parents[1]
SURROGATES = ("OvA", "ALCE")


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_arg, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("-ep", "--exper_name", "--name", dest="name")
    parser.add_argument("-m", "--model")
    parser.add_argument("-s", "--surrogate", choices=SURROGATES)
    parser.add_argument("-a", "--alpha", type=float)
    parser.add_argument("-r", "--random_seed", "--seed", dest="random_seed", type=int)
    parser.add_argument("-d", "--debug", choices=["true", "false"], type=str.lower, default="false")
    parser.add_argument("-v", "--val_split", type=float, default=0.1)
    parser.add_argument(
        "-mf", "--max_frac", type=float, default=1.0,
        help="Fraction of source-model baseline deferrals to retain.",
    )
    paper_training = ROOT / "configs" / "training.yaml"
    parser.add_argument(
        "--training-config", type=Path, default=paper_training if paper_training.is_file() else None,
        help="Optimizer and schedule settings (default: configs/training.yaml).",
    )
    parser.add_argument("--epochs", type=int, help="Override the surrogate training epochs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shifted-batch-size", type=int, default=4096)
    parser.add_argument("--expert-k", default="6", help="Comma-separated CIFAR expert sizes.")
    parser.add_argument(
        "--legacy-cifar-split", action="store_true",
        help="Use the historical CIFAR split (automatic for weights/cifar10_ova_k6_seed42.pt).",
    )
    parser.add_argument("--corruptions", help="Comma-separated corruption names.")
    parser.add_argument("--max-severity", type=int, help="Highest shift level to evaluate (0 for clean data only).")
    parser.add_argument("--annotator", default="random_annotator")
    project = Path(os.environ.get("L2D_PROJECT_ROOT", ROOT))
    paths = (
        ("data-dir", "L2D_DATA_DIR", "data"),
        ("results-dir", "L2D_RESULTS_DIR", "output/expers"),
        ("checkpoints-dir", "L2D_CHECKPOINTS_DIR", "checkpoints"),
        ("columns-path", "L2D_COLUMNS_PATH", "configs/columns.json"),
    )
    for flag, env, path in paths:
        default_path = Path(os.environ.get(env, project / path))
        parser.add_argument(f"--{flag}", type=Path, default=default_path)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate an OvA model state_dict directly.")
    parser.add_argument(
        "--paper-ham-backbone", action="store_true",
        help="Initialize a fresh HAM10000 fit from a local weights/ham10k_resnet34_100epochs_3407.pt.",
    )
    if config_arg.config:
        with config_arg.config.open() as source:
            settings = yaml.safe_load(source)
        # Older experiment configs named the fixed rule explicitly.
        if settings.pop("deferral_method", DEFERRAL_METHOD) != DEFERRAL_METHOD:
            parser.error(f"legacy deferral_method in config must be {DEFERRAL_METHOD}")
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    _resolve_and_validate_args(parser, args)
    return args


def _resolve_and_validate_args(parser, args):
    if args.config is None:
        parser.error("choose an experiment with --config configs/<experiment>.yaml")
    if args.surrogate not in SURROGATES:
        parser.error("conformal experiments require surrogate OvA or ALCE")
    expected_model = {"CIFAR10": "SimpleCNN", "HAM10000": "ResNet34", "HateSpeech": "LinearNet"}
    if args.model != expected_model[args.dataset]:
        parser.error(f"{args.dataset} uses {expected_model[args.dataset]}")
    if args.epochs is not None and (args.epochs < 1 or args.checkpoint):
        parser.error("--epochs requires a positive count for a training run")
    if args.checkpoint and args.surrogate != "OvA":
        parser.error("--checkpoint accepts an OvA state_dict")
    if args.legacy_cifar_split and args.dataset != "CIFAR10":
        parser.error("--legacy-cifar-split applies to CIFAR-10")
    bundled_cifar = ROOT / "weights" / "cifar10_ova_k6_seed42.pt"
    if (
        args.dataset == "CIFAR10" and args.checkpoint
        and args.checkpoint.resolve() == bundled_cifar.resolve()
    ):
        # Some images in the new calibration and test splits belonged to this model's training pool.
        args.legacy_cifar_split = True
    if args.paper_ham_backbone and (args.dataset != "HAM10000" or args.checkpoint):
        parser.error("--paper-ham-backbone applies to HAM10000 OvA or ALCE training")
    if not 0 < args.alpha < 1 or not 0 < args.val_split < 1:
        parser.error("alpha and val_split must lie strictly between 0 and 1")
    max_severity = 6
    if args.dataset == "CIFAR10" and args.evaluation != "sets":
        max_severity = 5
    if args.max_severity is None:
        args.max_severity = max_severity
    if not 0 <= args.max_severity <= max_severity:
        parser.error(f"max-severity must be between 0 and {max_severity} for this experiment")
    args.debug = args.debug == "true"
    args.expert_k = [int(k) for k in str(args.expert_k).split(",")]
    args.corruptions = args.corruptions.split(",")
    if not (0 < args.max_frac <= 1 and args.batch_size > 0 and args.shifted_batch_size > 0):
        parser.error("batch sizes must be positive and max_frac must be in (0, 1]")
    if args.dataset == "CIFAR10" and any(k < 0 or k > 10 for k in args.expert_k):
        parser.error("CIFAR expert sizes must be between 0 and 10")


def _seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _shift_levels(args):
    """In the CIFAR set sweep, SVHN is one target shared by all corruption families."""
    include_svhn = args.dataset == "CIFAR10" and args.max_severity == 6
    last_corruption_level = 5 if include_svhn else args.max_severity
    for corruption in args.corruptions:
        for severity in range(last_corruption_level + 1):
            yield corruption, severity
    if include_svhn:
        yield "svhn", 6


def _evaluate_shifts(args, result, output_dir, device, expert_k):
    """Fit once on the source domain and evaluate each severity with a shared test prediction."""
    from conformal.metrics import append_metrics_row, evaluate_baseline

    from .data import load_dataset
    from .evaluation import calibrate, evaluate
    from .models import load_or_train_surrogate

    _seed_everything(args.random_seed)
    source_data = load_dataset(args, 0, args.corruptions[0], expert_k, device)
    surrogate = load_or_train_surrogate(args, source_data, device)
    source_calibration = None

    for corruption, severity in _shift_levels(args):
        _seed_everything(args.random_seed + severity)
        data = source_data if severity == 0 else load_dataset(args, severity, corruption, expert_k, device)
        if args.dataset == "HAM10000" and data.source_key != source_data.source_key:
            raise RuntimeError(
                "HAM10000 source splits or expert weights changed during this run. "
                "Restart with the newly prepared data."
            )

        tags = {"expert_k": expert_k, "corruption": corruption, "severity": severity}
        # HAM10000 resamples pixel masks on access. Every policy must see the same inputs.
        test_result = surrogate.test(data.data_test_loader)
        is_svhn = args.dataset == "CIFAR10" and severity == 6
        if not is_svhn:
            baseline_metrics = evaluate_baseline(test_result, surrogate.name, max_frac=args.max_frac)
            result = append_metrics_row(
                result, method="source_model", dataset=data.dataset_name, model=args.model,
                surrogate=args.surrogate, alpha=args.alpha, metrics=baseline_metrics, **tags,
            )
        if source_calibration is None:
            source_calibration = calibrate(args, source_data, surrogate)
        result = evaluate(
            args, result, data, surrogate, source_calibration, "cali_by_non_shifted",
            tags, output_dir, test_result=test_result,
        )
        if severity:
            target_calibration = calibrate(args, data, surrogate)
            result = evaluate(
                args, result, data, surrogate, target_calibration, "cali_by_shifted",
                tags, output_dir, test_result=test_result,
            )
    return result


def run(args):
    import pandas as pd
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.legacy_cifar_split:
        logging.info("Using the historical CIFAR-10 split with the official 10,000-image test set.")
    with args.columns_path.open() as source:
        schema = json.load(source)
    columns = schema["columns"]
    if args.evaluation == "intervals":
        columns = columns + schema.get("interval_columns", [])

    output_dir = (
        args.results_dir / args.name / args.model / DEFERRAL_METHOD / args.surrogate
        / f"seed_{args.random_seed}" / f"alpha_{args.alpha}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(columns=columns)
    for expert_k in args.expert_k:
        result = _evaluate_shifts(args, result, output_dir, device, expert_k)

    csv_path = output_dir / "metrics.csv"
    result.to_csv(csv_path, columns=columns, index=False)
    print(csv_path)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    run(args)


if __name__ == "__main__":
    main()
