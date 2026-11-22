"""Conformal prediction for a learning-to-defer rejector."""

DEFERRAL_METHOD = "ConformalAndThreshold"


def validate_deferral_method(method):
    if method != DEFERRAL_METHOD:
        raise ValueError(f"Unsupported deferral method {method!r}. Use {DEFERRAL_METHOD}.")
