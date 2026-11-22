"""Image preprocessing shared by the CIFAR classifier and its shifted test sets."""

from torchvision import transforms


def cifar_image_transform(augment=False):
    """Normalize with the CIFAR channel statistics after optional training augmentation."""
    steps = [transforms.ToTensor()]
    if augment:
        steps.extend([
            transforms.Pad(4, padding_mode="reflect"),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    steps.append(
        transforms.Normalize(
            mean=[value / 255.0 for value in (125.3, 123.0, 113.9)],
            std=[value / 255.0 for value in (63.0, 62.1, 66.7)],
        )
    )
    return transforms.Compose(steps)
