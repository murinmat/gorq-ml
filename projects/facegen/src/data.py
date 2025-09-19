import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize
import lightning as L

TRAIN_RADARS = [
    'KDMX',
    'KUEX'
]
VAL_RADARS = [
    'KGLD'
]
BASE_PATH = '/data/datasets/celeba/images/'
BASE_AUGMENTATIONS = [
    Resize((256, 256))
]
TRAIN_AUGMENTATIONS = [
    RandomHorizontalFlip(),
]


@dataclass(kw_only=True)
class FacesDataset(Dataset):
    images: list[Path]
    augmentations: Callable[[torch.Tensor], torch.Tensor]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_data = np.array(Image.open(self.images[index])).swapaxes(0, 2).swapaxes(1, 2)
        img_data = np.nan_to_num(img_data, nan=0, neginf=0, posinf=0)
        return self.augmentations(torch.as_tensor(img_data / 255., dtype=torch.float32))


class FacesDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_images: list[Path],
        val_images: list[Path],
        batch_size: int,
        num_workers: int,
        base_augmentations: list[Callable[[torch.Tensor], torch.Tensor]],
        train_augmentations: list[Callable[[torch.Tensor], torch.Tensor]],
    ):
        super().__init__()  # make sure LightningDataModule is initialized

        self.train_images = train_images
        self.val_images = val_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_augmentations = base_augmentations
        self.train_augmentations = train_augmentations

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=FacesDataset(
                images=self.train_images,
                augmentations=Compose(self.base_augmentations + self.train_augmentations),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=FacesDataset(
                images=self.val_images,
                augmentations=Compose(self.base_augmentations),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def load_all_images() -> list[Path]:
    return list(Path(BASE_PATH).iterdir())
