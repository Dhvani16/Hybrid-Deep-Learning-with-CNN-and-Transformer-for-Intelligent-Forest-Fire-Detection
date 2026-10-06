import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ['no_fire', 'fire']


def _build_transform(split: str, img_size: int) -> A.Compose:
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FireDataset(Dataset):
    """Binary fire / no-fire image classification dataset.

    Expects directory layout:
        root_dir/
            train/fire/*.jpg   train/no_fire/*.jpg
            val/fire/*.jpg     val/no_fire/*.jpg
            test/fire/*.jpg    test/no_fire/*.jpg
    """

    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 224):
        self.transform = _build_transform(split, img_size)
        self.samples: list[tuple[str, int]] = []

        for label, cls in enumerate(CLASS_NAMES):
            folder = os.path.join(root_dir, split, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Expected directory not found: {folder}")
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(folder, fname), label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root_dir}/{split}/")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        img = self.transform(image=img)['image']
        return img, label

    def class_counts(self) -> dict:
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {CLASS_NAMES[k]: v for k, v in counts.items()}


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
) -> dict:
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = FireDataset(data_dir, split=split, img_size=img_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders
