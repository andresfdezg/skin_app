import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Affine(rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(size: int) -> A.Compose:
    """Resize the short side and take the central square, where the lesion usually is."""
    return A.Compose([
        A.SmallestMaxSize(max_size=size),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
