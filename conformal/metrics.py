"""Collect baseline routing metrics and format per-policy results for the experiment CSV."""

import numpy as np
import pandas as pd
import torch

from helpers.metrics import compute_deferral_metrics
from .scores import surrogate_probabilities


def mean_indicator(mask):
    """Mean of a boolean indicator. Empty subgroups have undefined rates (NaN)."""
    return mask.float().mean().item()


def append_metrics_row(
    result, method, dataset, model, surrogate, alpha, expert_k, corruption, severity, metrics,
):
    row = {
        "method": method,
        "corruption": corruption,
        "severity": severity,
        "expert_k": expert_k,
        "dataset": dataset,
        "model_name": model,
        "surrogate_fun": surrogate,
        "alpha": alpha,
        "confidence": 1 - alpha,
        **metrics,
    }
    return pd.concat([result, pd.DataFrame([row])], ignore_index=True)


def _baseline_entropy(test_result, surrogate_name):
    """Mean binary entropy in bits, as reported by the source-model baseline."""
    logits = torch.as_tensor(test_result["output_g_all"], dtype=torch.float32)
    expert_prob = surrogate_probabilities(logits, surrogate_name)[:, -1]
    distribution = torch.stack((expert_prob, 1 - expert_prob)).clamp(min=1e-12)
    return -(distribution * distribution.log2()).sum(dim=0).mean().item()


def _baseline_diagnostics(test_result, deferred, policy_metrics):
    """Historical optional columns computed on the realized baseline decisions."""
    expert_labels = test_result["hum_preds"]
    predictions = test_result["preds"]
    labels = test_result["labels"]
    expert_correct = expert_labels == labels
    expert_needed = expert_correct & (predictions != labels)
    expert_wrong = ~expert_correct

    # These legacy diagnostics intersect the routed integer class with a bit mask.
    # The paper's accuracy is computed separately using label equality.
    routed_class = predictions * (1 - deferred) + expert_labels * deferred
    class_bit = routed_class & expert_correct
    correct_deferrals = expert_correct & deferred
    return {
        "true_min_deferral_ratio": expert_correct.mean().item(),
        "true_deferred_ratio": (deferred == expert_correct).mean().item(),
        "rejector_nondefer_accuracy": (~deferred & expert_wrong).sum() / expert_wrong.sum(),
        "conservative_defer_acc": (deferred & expert_needed).sum() / expert_needed.sum(),
        "conservative_system_defer_acc": class_bit.sum() / expert_correct.sum(),
        "uncertainty_rejector_acc": policy_metrics["human_deferred_acc"],
        "uncertainty_classifier_acc": policy_metrics["classifier_nondeferred_acc"],
        "defer_system_acc": ((class_bit & deferred).sum() / deferred.sum()).item(),
        "uncertainty_system_acc": (class_bit & correct_deferrals).sum() / correct_deferrals.sum(),
    }


def evaluate_baseline(test_result, surrogate_name, *, max_frac=1):
    """Score the original L2D policy, optionally keeping a fraction of its deferrals.

    The CSV's accuracy follows the evaluated policy. Its optional system_acc field
    retains the upstream accuracy of the original surrogate routing decisions.
    """
    surrogate_metrics = compute_deferral_metrics(test_result)
    policy_metrics = surrogate_metrics
    deferred = test_result["defers"] == 1
    if max_frac < 1 and deferred.any():
        candidates = np.flatnonzero(deferred)
        selected = np.random.choice(candidates, int(max_frac * len(candidates)), replace=False)
        deferred[:] = False
        deferred[selected] = True
        policy_metrics = compute_deferral_metrics({**test_result, "defers": deferred})

    return {
        "accuracy": policy_metrics["system_acc"],
        "or_accuracy": None,
        "conformal_q": None,
        **_baseline_diagnostics(test_result, deferred, policy_metrics),
        "entropy": _baseline_entropy(test_result, surrogate_name),
        "conformal_width_array": None,
        "classifier_empirical_coverage": None,
        "empirical_coverage": None,
        "empirical_coverage_oracle": None,
        "empirical_width": None,
        "empirical_width_ratio": None,
        "num_defer": deferred.sum(),
        "ratio_defer": deferred.mean(),
        **surrogate_metrics,
    }
