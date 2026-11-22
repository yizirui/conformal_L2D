from pathlib import Path

import torch
import torch.nn.functional as F

from .expert_model import MLPMixer


EXPERT_WEIGHTS = Path(__file__).resolve().parents[1] / "MLP_Mixer_model" / "m_expert"


class MetadataExpert:
    """Annotate HAM images with the released Multi_L2D metadata expert."""

    def __init__(self, device, weights_path=EXPERT_WEIGHTS):
        self.expert = MLPMixer(
            image_size=224, patch_size=16, in_channels=3, num_features=128,
            expansion_factor=2, num_layers=8, num_classes=7,
        ).to(device)
        self.expert.load_state_dict(torch.load(weights_path, map_location=device))
        self.expert.eval()

    def MLPMixer(self, images, labels, metadata):
        probabilities = F.softmax(self.expert(images, metadata), dim=1)
        return probabilities.max(dim=1).indices
