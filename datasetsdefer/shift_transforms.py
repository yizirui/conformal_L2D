"""Pixel masking and Gaussian blur for the HAM10000 severity sweep."""

from functools import lru_cache

import numpy as np
from torchvision.transforms import GaussianBlur


class HideTransform:
    """Zero the same randomly selected pixels in every channel, in place."""

    def __init__(self, level=0):
        self.level = level
        self.noise_levels = {0: 0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.8}

    def __call__(self, img):
        height, width = img.shape[1:]
        total_pixels = height * width
        count = int(total_pixels * self.noise_levels.get(self.level, 0))
        # Sampling even the empty mask preserves the experiment's NumPy RNG sequence.
        indices = np.random.choice(total_pixels, count, replace=False)
        rows, cols = np.divmod(indices, width)
        img[:, rows, cols] = 0
        return img


@lru_cache(maxsize=6)
def _gaussian_blur(kernel_size, sigma):
    return GaussianBlur(kernel_size=kernel_size, sigma=sigma)


class GaussianBlurTransform:
    """Apply the Gaussian kernel and sigma for the configured severity."""

    def __init__(self, severity=0):
        self.severity = severity
        self.kernel_sizes = {s: (2 * s + 1, 2 * s + 1) for s in range(7)}
        self.sigma = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 6.0}

    def __call__(self, img):
        kernel_size = self.kernel_sizes.get(self.severity, (1, 1))
        if kernel_size == (1, 1):
            return img
        blur = _gaussian_blur(kernel_size, self.sigma[self.severity])
        return blur(img)
