import numpy as np
import torch
import datetime_glob
import h5py
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor, RandomVerticalFlip
from sklearn.model_selection import train_test_split


RADARS = ['KDMX', 'KLNX', 'KGLD']
BASE_AUGMENTATIONS = [
    ToTensor(),
    Resize((256, 256))
]
TRAIN_AUGMENTATIONS = [
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
]


@dataclass(kw_only=True)
class WeatherGenDataset(Dataset):
    images: list[Path]
    augmentations: Callable[[torch.Tensor], torch.Tensor]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        with h5py.File(self.images[index]) as f:
            data = np.clip(np.array(f['input/DBZH_pseudocappi'])[0, 0], a_min=0, a_max=1) * 2 - 1
        return self.augmentations(data)

    @staticmethod
    def get_train_dataset(path_format: str, train_size: float) -> 'WeatherGenDataset':
        all_images = []
        for r in RADARS:
            all_images += [x[1] for x in datetime_glob.walk(path_format.format(radar=r))]
        wanted_images = train_test_split(all_images, train_size=train_size, random_state=42)[0]
        return WeatherGenDataset(
            images=wanted_images,
            augmentations=Compose(BASE_AUGMENTATIONS + TRAIN_AUGMENTATIONS)
        )

    @staticmethod
    def get_val_dataset(path_format: str, train_size: float) -> 'WeatherGenDataset':
        all_images = []
        for r in RADARS:
            all_images += [x[1] for x in datetime_glob.walk(path_format.format(radar=r))]
        wanted_images = train_test_split(all_images, train_size=train_size, random_state=42)[1]
        return WeatherGenDataset(
            images=wanted_images,
            augmentations=Compose(BASE_AUGMENTATIONS)
        )
