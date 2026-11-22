"""Calibrate a rejector and collect metrics for one evaluation split."""

import json

import torch

from conformal.deferral_intervals import (
    calibrate_deferral_intervals,
    evaluate_deferral_intervals,
    predict_deferral_intervals,
)
from conformal.deferral_sets import (
    calibrate_deferral_sets,
    evaluate_deferral_sets,
    predict_deferral_sets,
)
from conformal.metrics import append_metrics_row


def calibrate(args, data, surrogate):
    """Fit the selected conformal method on the held-out calibration split."""
    loader = data.data_val_loader
    if args.evaluation == "intervals":
        return calibrate_deferral_intervals(loader, surrogate, args.alpha)
    return calibrate_deferral_sets(loader, surrogate, args.alpha)


def _cifar_expert_probability(labels, expert_k, device):
    """The expert knows the first k classes and guesses uniformly among ten otherwise."""
    labels = torch.as_tensor(labels, device=device)
    probabilities = torch.full_like(labels, 0.1, dtype=torch.float64)
    probabilities[labels < expert_k] = 1.0
    return probabilities


def evaluate(args, result, data, surrogate, calibration, method, tags, out, *, test_result=None):
    """Score one calibration on the test predictions and save its per-example outputs."""
    loader = data.data_test_loader
    if test_result is None:
        test_result = surrogate.test(loader)
    stem = f"k{tags['expert_k']}_{tags['corruption']}_s{tags['severity']}_{method}"

    if args.evaluation == "intervals":
        defer_to_expert, lower, upper, classifier_predictions = predict_deferral_intervals(
            loader, surrogate, args.alpha, *calibration, test_result=test_result,
        )
        true_expert_probability = None
        if args.dataset == "CIFAR10":
            true_expert_probability = _cifar_expert_probability(
                test_result["labels"], tags["expert_k"], lower.device,
            )
        metrics, examples = evaluate_deferral_intervals(
            test_result,
            defer_to_expert=defer_to_expert,
            classifier_predictions=classifier_predictions,
            lower_bound=lower,
            upper_bound=upper,
            surrogate_name=surrogate.name,
            true_expert_probability=true_expert_probability,
        )
        metadata = {
            "method_name": method,
            "dataset_name": data.dataset_name,
            **tags,
            "model_name": args.model,
        }
        with (out / f"{stem}.json").open("w") as destination:
            json.dump({**metadata, **examples}, destination)
    else:
        sets, classifier_predictions, entropy = predict_deferral_sets(
            loader, surrogate, calibration, test_result=test_result,
        )
        sets_path = out / f"{stem}.pt" if args.dataset == "CIFAR10" else None
        metrics = evaluate_deferral_sets(
            test_result, sets, classifier_predictions,
            threshold=calibration, entropy=entropy, save_path=sets_path,
        )

    return append_metrics_row(
        result, method=method, dataset=data.dataset_name, model=args.model,
        surrogate=args.surrogate, alpha=args.alpha, metrics=metrics, **tags,
    )
