"""Routing and evaluation of the deferral intervals in Section 3.3."""

import torch

from helpers.metrics import compute_deferral_metrics

from .conformal_regression import calibrate_deferral_intervals, predict_deferral_intervals
from .metrics import mean_indicator
from .scores import device, surrogate_probabilities

__all__ = ["calibrate_deferral_intervals", "predict_deferral_intervals", "evaluate_deferral_intervals"]


def _interval_inclusion(lower, upper, probability):
    """Fraction of examples whose closed interval contains the supplied probability."""
    return ((lower <= probability) & (probability <= upper)).double().mean().item()


def _true_expert_coverage(lower, upper, probability):
    """Section 3.3 measures coverage of the true probability of expert correctness."""
    if probability is None:
        return None, float("nan")

    probability = torch.as_tensor(probability, dtype=torch.float64, device=lower.device)
    if probability.shape != lower.shape:
        raise ValueError("Expected one true expert probability per test example.")
    if not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("True expert probabilities must be finite and in [0, 1].")
    return probability, _interval_inclusion(lower, upper, probability)


def _conditional_rate(event, condition):
    """Ratio of matching counts, with NaN when no example meets the condition."""
    return ((event & condition).float().sum() / condition.float().sum()).item()


def _interval_routing_metrics(
    *, defer, answered, consensus_answered, preferred_defers,
    classifier_correct, expert_correct, base_defers, base_classifier_correct,
):
    """Score each interval workflow on its own answered examples and routes."""
    interval_correct = torch.where(defer, expert_correct, classifier_correct)
    base_correct = torch.where(base_defers.bool(), expert_correct, base_classifier_correct)
    model_preferred_correct = torch.where(preferred_defers, expert_correct, classifier_correct)
    expert_only_correct = expert_correct & ~base_classifier_correct
    expert_correct_on_deferrals = expert_correct & defer

    num_defer = defer.sum().item()
    classifier_accuracy = mean_indicator(classifier_correct[~defer])
    preferred_expert_accuracy = mean_indicator(expert_correct[preferred_defers])
    return {
        "num_defer": num_defer,
        "ratio_defer": num_defer / len(defer),
        # Overall accuracy uses the full test set and the interval route.
        "accuracy": mean_indicator(interval_correct),
        "or_accuracy": mean_indicator(classifier_correct | expert_correct),
        "conformal_classifier_non_deferred_accuracy": classifier_accuracy,
        "conformal_human_deferred_accuracy": mean_indicator(expert_correct[defer]),
        "empirical_coverage_oracle": mean_indicator(defer | base_classifier_correct),
        # Compare the interval and base routes on exactly the same answered examples.
        "non_abstention_acc": mean_indicator(interval_correct[answered]),
        "non_abstention_base": mean_indicator(base_correct[answered]),
        "non_abstention_uncertainty_rejector_acc": mean_indicator(expert_correct[answered & defer]),
        "non_abstention_uncertainty_classifier_acc": mean_indicator(classifier_correct[answered & ~defer]),
        # Expert accuracy conditions on the subset routed to the expert.
        "model_preferred_uncertainty_rejector_acc": preferred_expert_accuracy,
        "model_preferred_uncertainty_classifier_acc": mean_indicator(classifier_correct[~preferred_defers]),
        "non_abstention_ratio": mean_indicator(answered),
        "non_abstention_deferral_ratio": mean_indicator(defer[answered]),
        "consensus_ratio": mean_indicator(consensus_answered),
        "consensus_acc": mean_indicator(interval_correct[consensus_answered]),
        "consensus_deferral_ratio": mean_indicator(defer[consensus_answered]),
        # Model-preferred system accuracy includes both human and classifier predictions.
        "model_preferred_acc": mean_indicator(model_preferred_correct),
        "model_preferred_deferral_ratio": mean_indicator(preferred_defers),
        "non_abstention_base_deferral_ratio": mean_indicator(base_defers[answered]),
        "true_min_deferral_ratio": mean_indicator(base_classifier_correct),
        "true_deferred_ratio": mean_indicator(defer & base_classifier_correct),
        "min_defer_policy_acc": mean_indicator(defer[base_classifier_correct]),
        "rejector_nondefer_accuracy": _conditional_rate(~defer, ~expert_correct),
        "conservative_defer_acc": _conditional_rate(defer, expert_only_correct),
        "conservative_system_defer_acc": _conditional_rate(interval_correct, expert_correct),
        "uncertainty_rejector_acc": _conditional_rate(expert_correct, defer),
        "uncertainty_classifier_acc": classifier_accuracy,
        "uncertainty_system_acc": _conditional_rate(interval_correct, expert_correct_on_deferrals),
        "defer_system_acc": ((interval_correct & defer).sum() / defer.sum()).item(),
    }


def evaluate_deferral_intervals(
    test_result,
    *,
    defer_to_expert,
    classifier_predictions,
    lower_bound,
    upper_bound,
    surrogate_name,
    true_expert_probability=None,
    abstention_threshold=None,
):
    """Return workflow metrics and per-example decisions for one test prediction snapshot.

    Each policy is evaluated against the same test labels and expert predictions.
    The default abstention workflow withholds predictions whose interval contains
    0.5. A supplied width threshold instead withholds intervals wider than that value.
    """
    labels = torch.as_tensor(test_result["labels"], device=device)
    expert_labels = torch.as_tensor(test_result["hum_preds"], device=device)
    base_predictions = torch.as_tensor(test_result["preds"], device=device)
    base_defers = torch.as_tensor(test_result["defers"], device=device)
    logits = torch.as_tensor(test_result["output_g_all"], device=device)
    probabilities = surrogate_probabilities(logits, surrogate_name)
    fitted_expert_probability = probabilities[:, -1]

    defer = defer_to_expert.bool()
    valid = (
        (0 <= lower_bound) & (lower_bound < 1) & (0 < upper_bound)
        & (upper_bound <= 1) & (lower_bound <= upper_bound)
    )
    if not valid.all():
        raise ValueError("Expected ordered deferral intervals with lower in [0, 1) and upper in (0, 1].")
    width = upper_bound - lower_bound
    if abstention_threshold is None:
        abstain = (lower_bound <= 0.5) & (0.5 <= upper_bound)
    else:
        abstain = width > abstention_threshold
    consensus_answered = ~abstain | (classifier_predictions == expert_labels)
    full_interval = (lower_bound == 0) & (upper_bound == 1)
    preferred_defers = defer & ~full_interval
    metrics = _interval_routing_metrics(
        defer=defer,
        answered=~abstain,
        consensus_answered=consensus_answered,
        preferred_defers=preferred_defers,
        classifier_correct=(classifier_predictions == labels),
        expert_correct=(expert_labels == labels),
        base_defers=base_defers,
        base_classifier_correct=(base_predictions == labels),
    )
    true_expert_probability, coverage = _true_expert_coverage(
        lower_bound, upper_bound, true_expert_probability
    )
    fitted_score_inclusion = _interval_inclusion(
        lower_bound, upper_bound, fitted_expert_probability
    )
    mean_width = width.mean().item()

    metrics.update({
        "classifier_empirical_coverage": None,
        "empirical_coverage": coverage,
        "empirical_width": mean_width,
        "empirical_width_ratio": mean_width,  # The probability space [0, 1] has unit width.
        "conformal_q": 0,
        "fitted_score_inclusion": fitted_score_inclusion,
        # Keep the original rejector's system_acc alongside the interval policy metrics.
        **compute_deferral_metrics(test_result),
    })
    examples = {
        "labels": labels.tolist(),
        "test_pred": base_predictions.tolist(),
        "human_labels": expert_labels.tolist(),
        "defer_binary": defer.tolist(),
        "abstention": abstain.tolist(),
        "consensus_answered": consensus_answered.tolist(),
        "model_preferred_defers": preferred_defers.tolist(),
        "most_prob_arg": classifier_predictions.tolist(),
        "prob_y_x": fitted_expert_probability.tolist(),
        "true_expert_probability": (
            None if true_expert_probability is None else true_expert_probability.tolist()
        ),
        "image_prob_max": probabilities[:, :-1].max(dim=1).values.tolist(),
        "empty_interval": (lower_bound > upper_bound).tolist(),
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist(),
    }
    return metrics, examples
