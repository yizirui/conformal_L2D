"""Barber binary-regression calibration and interval construction for expert correctness."""

import math

import numpy as np
import torch
from scipy.optimize import minimize

from . import DEFERRAL_METHOD, validate_deferral_method
from .metrics import append_metrics_row
from .scores import device, surrogate_probabilities


def _bin_indices(scores, M):
    # Barber (13) uses [0, 1/M), ..., [(M-1)/M, 1].
    boundaries = torch.arange(1, M, device=scores.device, dtype=scores.dtype) / M
    return torch.bucketize(scores.contiguous(), boundaries, right=True)


def partition_indices(norm_rejector: torch.Tensor, M: int):
    """Return the calibration sample indices in each equal-width score bin."""
    assigned = _bin_indices(norm_rejector, M)
    return [torch.where(assigned == i)[0] for i in range(M)]


def upper_bound_region_prob(p_hat: torch.Tensor, n: int, alpha: float):
    """Upper-bound each bin's probability mass using Barber (2020), Eq. (9)."""
    correction = 3 * math.log(4 * len(p_hat) * n / alpha) / n
    return p_hat + torch.sqrt(p_hat * correction) + correction


def shrink_label_prob(pi_hat: torch.Tensor, p_hat: torch.Tensor, n: int, alpha: float):
    """Shrink each bin's correctness frequency toward one half using Barber, Eq. (10)."""
    counts = n * p_hat
    correction = 2 * math.log(4 * len(p_hat) * n / alpha) / counts.clamp(min=1)
    distance_to_boundary = torch.minimum(pi_hat, 1 - pi_hat)
    delta = torch.sqrt(distance_to_boundary * correction) + correction
    half = torch.full_like(pi_hat, 0.5)
    shrunk = torch.where(
        pi_hat <= 0.5, (pi_hat + delta).clamp(max=0.5), (pi_hat - delta).clamp(min=0.5)
    )
    return torch.where(counts > 0, shrunk, half)


def ell_scalar(t: float, a: float) -> float:
    """Expected randomized interval length from Barber (2020), section 2.2."""
    t = min(t, 1 - t)
    if t == 0:
        return 0.0
    if a >= 0.5:
        return 2 * (1 - a) * t
    if a >= t:
        return t / (2 * a)
    return 1 - a / (2 * t)


ell_vec = np.vectorize(ell_scalar, otypes=[float])


def allocate_bin_error_budgets(tilde_p: torch.Tensor, tilde_pi: torch.Tensor, alpha: float) -> np.ndarray:
    """Minimize expected interval length subject to Barber's miscoverage budget in Eq. (11)."""
    masses = tilde_p.detach().cpu().numpy().astype(float)
    rates = tilde_pi.detach().cpu().numpy().astype(float)

    def objective(a):
        return np.dot(masses, ell_vec(rates, a))

    constraint = {
        "type": "ineq",
        "fun": lambda a: alpha - masses @ a,
        "jac": lambda a: -masses,
    }
    initial = np.full_like(masses, min(1.0, alpha / masses.sum()))
    result = minimize(
        objective, initial, method="SLSQP", constraints=constraint,
        bounds=[(0.0, 1.0)] * len(masses), options={"ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Could not allocate interval error budgets: {result.message}")
    budgets = np.clip(result.x, 0, 1)
    spent = masses @ budgets
    if spent > alpha:
        # Eliminate floating-point constraint slack on the boundary.
        budgets *= alpha / spent
    return budgets


# Historical name for the optimizer over the per-bin error budgets a_m.
solve_a = allocate_bin_error_budgets


def randomized_intervals_tensor(tilde_pi: torch.Tensor, a_m: torch.Tensor):
    """Invert f(s) >= U in Barber, Eqs. (4)–(5). Reversed endpoints (1, 0) encode an empty set."""
    draw = torch.rand_like(tilde_pi)
    # For a rate at or below 1/2 the nonempty interval starts at zero.
    # Handle larger rates by reflecting the same construction around 1/2.
    tail_rate = torch.minimum(tilde_pi, 1 - tilde_pi)
    right = torch.ones_like(tail_rate)
    empty = ((tail_rate == 0) | (tail_rate == 0.5)) & (draw > 1 - a_m)
    right[(tail_rate == 0) & (draw > 0)] = 0

    interior = (tail_rate > 0) & (tail_rate < 0.5) & (draw > 0)
    high_budget = interior & (a_m >= 0.5)
    peak = 2 * (1 - a_m)
    empty |= high_budget & (draw > peak)
    nonempty_high_budget = high_budget & ~empty
    right[nonempty_high_budget] = 2 * tail_rate[nonempty_high_budget] * (
        1 - draw[nonempty_high_budget] / peak[nonempty_high_budget]
    )
    low_budget = interior & (a_m > 0) & (a_m < 0.5)
    right[low_budget] = (
        (1 - draw[low_budget]) * tail_rate[low_budget] / a_m[low_budget]
    ).clamp(max=1)

    lower = torch.where(tilde_pi <= 0.5, torch.zeros_like(tail_rate), 1 - right)
    upper = torch.where(tilde_pi <= 0.5, right, torch.ones_like(tail_rate))
    return lower.masked_fill(empty, 1), upper.masked_fill(empty, 0)


def get_confidence_intervals(test_scores, tilde_pi, a_m, M):
    """Draw a raw Barber set for each test observation, including possible empty draws."""
    bins = _bin_indices(test_scores, M)
    return randomized_intervals_tensor(tilde_pi[bins], a_m[bins])


def _probabilities(test_result, surrogate):
    logits = torch.as_tensor(test_result["output_g_all"], device=device)
    return surrogate_probabilities(logits, surrogate.name)


def calibrate_deferral_intervals(dataloader, surrogate, alpha):
    """Estimate conservative region masses and correctness rates on calibration data."""
    calibration = surrogate.test(dataloader)
    expert_scores = _probabilities(calibration, surrogate)[:, -1]
    n_calibration = expert_scores.numel()
    if n_calibration == 0:
        raise ValueError("Binary-regression calibration requires at least one observation.")
    # Barber §3.2.1 chooses the bin count for two equally sized halves.
    # Only the calibration half supplies observations for Eqs. (9) and (10).
    total = 2 * n_calibration
    num_bins = math.ceil(math.sqrt(total / math.log(total)))
    bins = partition_indices(expert_scores, num_bins)
    expert_labels = torch.as_tensor(calibration["hum_preds"], device=device)
    labels = torch.as_tensor(calibration["labels"], device=device)
    expert_correct = expert_labels == labels
    bin_counts = torch.tensor([len(indices) for indices in bins], device=device, dtype=torch.float64)
    correct_counts = torch.stack([expert_correct[indices].sum() for indices in bins])
    bin_masses = bin_counts / n_calibration
    correctness_rates = torch.where(bin_counts > 0, correct_counts / bin_counts.clamp(min=1), 0.5)
    mass_bounds = upper_bound_region_prob(bin_masses, n_calibration, alpha)
    shrunk_rates = shrink_label_prob(correctness_rates, bin_masses, n_calibration, alpha)
    return mass_bounds, shrunk_rates


def predict_deferral_intervals(
    dataloader, surrogate, alpha, tilde_p, tilde_pi, *, test_result=None
):
    """Construct intervals and defer when the lower endpoint exceeds classifier confidence."""
    budgets = torch.as_tensor(
        allocate_bin_error_budgets(tilde_p, tilde_pi, alpha), dtype=tilde_pi.dtype, device=tilde_pi.device
    )
    if test_result is None:
        test_result = surrogate.test(dataloader)
    probs = _probabilities(test_result, surrogate)
    lower, upper = get_confidence_intervals(probs[:, -1], tilde_pi, budgets, len(budgets))
    # Barber can draw the empty set. Section 3.3 needs an interval; [0, 1]
    # conservatively includes every possible expert-correctness probability.
    empty = lower > upper
    lower = lower.masked_fill(empty, 0)
    upper = upper.masked_fill(empty, 1)
    # For singletons at 0 or 1, widen by one representable step into [0, 1].
    zero, one = lower.new_tensor(0), lower.new_tensor(1)
    lower = lower.clamp(max=torch.nextafter(one, zero))
    upper = upper.clamp(min=torch.nextafter(zero, one))
    classifier_confidence = probs[:, :-1].max(dim=1).values
    defer = lower > classifier_confidence
    predictions = torch.as_tensor(test_result["preds"], device=device)
    return defer, lower, upper, predictions


def evaluate_conformal_regression(
    result,
    method_name,
    dataset_name,
    Surrogate,
    prediction_set_indictor,
    most_prob_arg,
    dataloader,
    model_name,
    surgurate_fun_name,
    alpha,
    lower_bound,
    upper_bound,
    abstention_threshold=None,
    deferral_method=DEFERRAL_METHOD,
    expert_k=None,
    corruption=None,
    severity=None,
    *,
    test_result=None,
    true_expert_probability=None,
):
    """Compatibility wrapper for the original DataFrame-based evaluation API."""
    from .deferral_intervals import evaluate_deferral_intervals

    validate_deferral_method(deferral_method)
    if test_result is None:
        test_result = Surrogate.test(dataloader)
    metrics, examples = evaluate_deferral_intervals(
        test_result,
        defer_to_expert=prediction_set_indictor,
        classifier_predictions=most_prob_arg,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        surrogate_name=Surrogate.name,
        true_expert_probability=true_expert_probability,
        abstention_threshold=abstention_threshold,
    )
    result = append_metrics_row(
        result, method_name, dataset_name, model_name, surgurate_fun_name, alpha,
        expert_k, corruption, severity, metrics,
    )
    return result, {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "expert_k": expert_k,
        "corruption": corruption,
        "severity": severity,
        "model_name": model_name,
        **examples,
    }
