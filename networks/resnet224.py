import torch
import torchvision


class ResNet34(torch.nn.Module):
    def __init__(self, weights="imagenet"):
        """Initialize from ImageNet, a backbone state_dict path, or random weights (None)."""
        super().__init__()

        imagenet = weights == "imagenet"
        self.resnet = torchvision.models.resnet34(pretrained=imagenet)
        self.n_features = self.resnet.fc.in_features
        del self.resnet.fc
        if weights is not None and not imagenet:
            self.load_state_dict(torch.load(weights, map_location="cpu"))

        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Hemmer et al ===
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)

        features = torch.flatten(x, 1)
        return features
