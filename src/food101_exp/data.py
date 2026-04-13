from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class Food101Paths:
    images_root: Path
    train_list: Path
    test_list: Path


class Food101FromMeta(Dataset):
    def __init__(
        self,
        list_path: str | Path,
        images_root: str | Path,
        transform=None,
        max_samples: int | None = None,
        class_to_idx: dict[str, int] | None = None,
    ):
        self.list_path = Path(list_path)
        self.images_root = Path(images_root)
        self.transform = transform
        self.samples = self.list_path.read_text().splitlines()
        if max_samples is not None:
            self.samples = select_samples(self.samples, max_samples)
        if class_to_idx is None:
            classes = sorted({sample.split("/")[0] for sample in self.samples})
            self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path = self.samples[index]
        class_name = rel_path.split("/")[0]
        image_path = self.images_root / f"{rel_path}.jpg"
        image = Image.open(image_path).convert("RGB")
        target = self.class_to_idx[class_name]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(image_size: int):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    paths: Food101Paths,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
):
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = Food101FromMeta(paths.train_list, paths.images_root, train_transform, max_train_samples)
    test_dataset = Food101FromMeta(
        paths.test_list,
        paths.images_root,
        eval_transform,
        max_test_samples,
        train_dataset.class_to_idx,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, test_dataset, train_loader, test_loader


def select_samples(samples: list[str], max_samples: int) -> list[str]:
    if max_samples >= len(samples):
        return samples
    if max_samples <= 1:
        return [samples[0]]
    step = (len(samples) - 1) / (max_samples - 1)
    indices = [round(index * step) for index in range(max_samples)]
    return [samples[index] for index in indices]
