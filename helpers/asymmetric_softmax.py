"""Asymmetric softmax for the classifier and expert in Equation (6)."""
import torch


def expert_log_odds(scorer):
    """Log odds of expert correctness after excluding one highest-scoring class."""
    class_scores = scorer[:, :-1]
    expert_score = scorer[:, -1]
    if class_scores.shape[1] == 1:
        # Only the expert term remains in the denominator of Equation (6).
        return torch.full_like(expert_score, torch.inf)
    highest_class = class_scores.argmax(dim=1, keepdim=True)
    others = class_scores.scatter(1, highest_class, -torch.inf)
    # Center first so a large common offset cannot swallow the log sum.
    offset = others.max(dim=1).values.detach()
    return (expert_score - offset) - torch.logsumexp(others - offset[:, None], dim=1)


def asymmetric_softmax(scorer):
    class_probs = scorer[:, :-1].softmax(dim=1)
    expert_prob = expert_log_odds(scorer).sigmoid()
    return torch.cat((class_probs, expert_prob[:, None]), dim=1)
