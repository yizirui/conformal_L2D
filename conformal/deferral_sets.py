"""Calibrate deferral sets for expert correctness and evaluate the workflows in §3.2."""

from math import ceil
from pathlib import Path

import torch

from helpers.metrics import compute_deferral_metrics

from .metrics import mean_indicator
from .scores import device, surrogate_probabilities


def _tensor_from_result(result_dict, key, dtype):
    values = result_dict.get(key)
    return None if values is None else torch.as_tensor(values, dtype=dtype, device=device)


def expert_nonconformity(labels, human_labels, logits, surrogate_name):
    """Return one minus the probability of the observed expert-correctness label."""
    expert_prob = surrogate_probabilities(logits, surrogate_name)[:, -1]
    return torch.where(labels == human_labels, 1 - expert_prob, expert_prob)


def calibrate_deferral_sets(dataloader, surrogate, alpha):
    """Compute the split-conformal quantile for the observed expert correctness labels."""
    result = surrogate.test(dataloader)
    labels = _tensor_from_result(result, "labels", torch.long)
    expert_labels = _tensor_from_result(result, "hum_preds", torch.long)
    logits = _tensor_from_result(result, "output_g_all", torch.float32)
    if labels is None or expert_labels is None or logits is None:
        raise ValueError("Surrogate.test must return labels, human labels, and logits.")

    scores = expert_nonconformity(labels, expert_labels, logits, surrogate.name)
    n = scores.numel()
    if n == 0:
        raise ValueError("Deferral-set calibration requires at least one observation.")
    rank = ceil((n + 1) * (1 - alpha))
    # All scores are bounded by 1. At rank n + 1 the full set admits both labels.
    if rank > n:
        qhat = scores.new_tensor(1.0)
    else:
        qhat = scores.kthvalue(max(1, rank)).values
    torch.cuda.empty_cache()
    return qhat


def predict_deferral_sets(dataloader, surrogate, qhat, *, test_result=None):
    """Return correctness sets, classifier predictions, and Bernoulli entropy in nats."""
    if test_result is None:
        test_result = surrogate.test(dataloader)
    elif test_result.get("output_g_all") is None:
        raise ValueError("A supplied test_result must include output_g_all to build conformal sets.")

    logits = _tensor_from_result(test_result, "output_g_all", torch.float32)
    if logits is None:
        with torch.no_grad():
            logits = torch.cat([surrogate.model(images.to(device)).detach() for images, _, _ in dataloader])

    predictions = _tensor_from_result(test_result, "preds", torch.long)
    if predictions is None:
        predictions = logits[:, :-1].argmax(dim=1)

    expert_prob = surrogate_probabilities(logits, surrogate.name)[:, -1]
    # Compare the same nonconformity scores used in calibration so ties are included.
    include_incorrect = expert_prob <= qhat
    include_correct = (1 - expert_prob) <= qhat
    # Equation (8) returns the full set when neither outcome clears the threshold.
    neither = ~(include_incorrect | include_correct)
    sets = torch.stack((include_incorrect | neither, include_correct | neither), dim=1)

    probabilities = torch.stack((expert_prob, 1 - expert_prob)).clamp(min=1e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=0)
    return sets, predictions, entropy


def _set_policy_metrics(
    sets, expert_index, classifier_predictions, labels, expert_labels,
    base_predictions, base_defers, entropy,
):
    """Evaluate the fixed set policy and the abstention, consensus, and preference workflows."""
    # Columns index the two possible outcomes of expert correctness (0 and 1).
    contains_incorrect = sets[:, 0].bool()
    contains_correct = sets[:, expert_index].bool()
    expert_correct = expert_labels == labels
    classifier_correct = classifier_predictions == labels
    base_classifier_correct = base_predictions == labels

    base_defers = base_defers.bool()
    fixed_policy_defers = contains_correct & base_defers
    fixed_policy_correct = torch.where(fixed_policy_defers, expert_correct, classifier_correct)
    base_correct = torch.where(base_defers, expert_correct, base_classifier_correct)
    expert_outcome_covered = torch.where(expert_correct, contains_correct, contains_incorrect)
    mean_set_size = sets.float().sum(dim=1).mean().item()

    uncertain = contains_incorrect & contains_correct
    abstention_answered = ~uncertain
    disagreement = classifier_predictions != expert_labels
    consensus_answered = ~(uncertain & disagreement)
    # In §3.2 the base rejector handles singletons. Either route gives the same label on agreement.
    workflow_correct = torch.where(base_defers, expert_correct, classifier_correct)
    # Appendix A.2 assigns an uncertain agreement to the classifier for counting deferrals.
    human_preferred_defers = torch.where(uncertain, disagreement, base_defers)
    model_preferred_defers = abstention_answered & base_defers
    human_preferred_correct = torch.where(human_preferred_defers, expert_correct, classifier_correct)
    model_preferred_correct = torch.where(model_preferred_defers, expert_correct, classifier_correct)
    correct_expert_in_set = contains_correct & expert_correct

    n_deferred = fixed_policy_defers.sum().item()
    classifier_accuracy = mean_indicator(classifier_correct[~fixed_policy_defers])
    expert_accuracy = mean_indicator(expert_correct[fixed_policy_defers])
    abstention_deferral_ratio = mean_indicator(base_defers[abstention_answered])
    consensus_deferral_ratio = mean_indicator(base_defers[consensus_answered])
    return {
        "non_abstention_acc": mean_indicator(workflow_correct[abstention_answered]),
        "non_abstention_base": mean_indicator(base_correct[abstention_answered]),
        "uncertainty_rejector_acc": expert_accuracy,
        "uncertainty_classifier_acc": classifier_accuracy,
        "non_abstention_uncertainty_rejector_acc": mean_indicator(
            expert_correct[abstention_answered & base_defers]
        ),
        "non_abstention_uncertainty_classifier_acc": mean_indicator(
            classifier_correct[abstention_answered & ~base_defers]
        ),
        "consensus_uncertainty_rejector_acc": mean_indicator(expert_correct[consensus_answered & base_defers]),
        "consensus_uncertainty_classifier_acc": mean_indicator(classifier_correct[consensus_answered & ~base_defers]),
        "human_preferred_uncertainty_rejector_acc": mean_indicator(expert_correct[human_preferred_defers]),
        "human_preferred_uncertainty_classifier_acc": mean_indicator(classifier_correct[~human_preferred_defers]),
        "model_preferred_acc": mean_indicator(model_preferred_correct),
        "model_preferred_deferral_ratio": mean_indicator(model_preferred_defers),
        "model_preferred_uncertainty_rejector_acc": mean_indicator(expert_correct[model_preferred_defers]),
        "model_preferred_uncertainty_classifier_acc": mean_indicator(classifier_correct[~model_preferred_defers]),
        "entropy": entropy.mean().item(),
        "consensus_acc": mean_indicator(workflow_correct[consensus_answered]),
        "consensus_base_acc": mean_indicator(base_correct[consensus_answered]),
        "non_abstention_ratio": mean_indicator(abstention_answered),
        "consensus_ratio": mean_indicator(consensus_answered),
        "non_abstention_deferral_ratio": abstention_deferral_ratio,
        "consensus_deferral_ratio": consensus_deferral_ratio,
        "human_preferred_acc": mean_indicator(human_preferred_correct),
        "human_preferred_deferral_ratio": mean_indicator(human_preferred_defers),
        "human_preferred_query_ratio": mean_indicator(uncertain | base_defers),
        "non_abstention_base_deferral_ratio": abstention_deferral_ratio,
        "consensus_base_deferral_ratio": consensus_deferral_ratio,
        "non_abstention_conformal_coverage": mean_indicator(expert_outcome_covered[abstention_answered]),
        "consensus_conformal_coverage": mean_indicator(expert_outcome_covered[consensus_answered]),
        "true_min_deferral_ratio": mean_indicator(correct_expert_in_set),
        "true_deferred_ratio": mean_indicator(fixed_policy_defers & correct_expert_in_set),
        "accuracy": mean_indicator(fixed_policy_correct),
        "or_accuracy": mean_indicator(classifier_correct | correct_expert_in_set),
        "conformal_classifier_non_deferred_accuracy": classifier_accuracy,
        "conformal_human_deferred_accuracy": expert_accuracy,
        "classifier_empirical_coverage": mean_indicator(contains_incorrect & base_classifier_correct),
        "empirical_coverage": mean_indicator(expert_outcome_covered),
        "empirical_coverage_oracle": mean_indicator(~fixed_policy_defers | expert_correct),
        "empirical_width": mean_set_size,
        "empirical_width_ratio": mean_set_size / sets.shape[1],
        "num_defer": n_deferred,
        "ratio_defer": n_deferred / len(sets),
    }


def evaluate_deferral_sets(
    test_result, sets, classifier_predictions, *, threshold, entropy, expert_index=1, save_path=None,
):
    """Score the set policies on one test snapshot and optionally save the per-example sets."""
    labels = torch.as_tensor(test_result["labels"], device=device)
    expert_labels = torch.as_tensor(test_result["hum_preds"], device=device)
    metrics = _set_policy_metrics(
        sets=sets,
        expert_index=expert_index,
        classifier_predictions=classifier_predictions,
        labels=labels,
        expert_labels=expert_labels,
        base_predictions=torch.as_tensor(test_result["preds"], device=device),
        base_defers=torch.as_tensor(test_result["defers"], device=device),
        entropy=entropy,
    )
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "conformal_sets": sets,
                "labels": labels,
                "human_labels": expert_labels,
                "preds": test_result["preds"],
            },
            save_path,
        )

    metrics["conformal_q"] = float(threshold)
    metrics.update(compute_deferral_metrics(test_result))
    return metrics
