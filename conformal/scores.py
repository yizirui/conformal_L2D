"""Probability estimates shared by the deferral-set and interval methods."""

import torch

from helpers.asymmetric_softmax import asymmetric_softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def surrogate_probabilities(logits, surrogate_name):
    """Map logits to class and expert-correctness probabilities using Equations (5) and (6)."""
    if surrogate_name == "OvA":
        return torch.sigmoid(logits)
    if surrogate_name in ("ALCE", "A-SM", "AsymLceSurrogate", "AsymLce"):
        return asymmetric_softmax(logits)
    raise ValueError(f"Unknown conformal surrogate: {surrogate_name}")
