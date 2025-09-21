import numpy as np
import torch
import datetime_glob
import os
import h5py
from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor, RandomVerticalFlip
from lightning import LightningDataModule


RADARS = ['KDMX', 'KLNX', 'KGLD']
PATH_FORMAT = '/mnt/remote/mp_data_cmuchalek/nexrad/experiment-cache-hdf5/{radar}/%Y/%m/%d/{radar}_%Y-%m-%d_%H-%M-%S.h5'
BASE_AUGMENTATIONS = [
    ToTensor(),
    Resize(256)
]
TRAIN_AUGMENTATIONS = [
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
]


@dataclass(kw_only=True)
class WeatherGenDataset(Dataset):
    files: list[Path]
    augmentations: Callable[[torch.Tensor], torch.Tensor]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with h5py.File(self.files[index]) as f:
            data = np.clip(np.array(f['input/DBZH_pseudocappi'])[0, 0] / 60., a_min=0, a_max=1)
        return self.augmentations(data)


class WeatherGenDataModule(LightningDataModule):
    def __init__(
        self,
        train_files: list[Path],
        val_files: list[Path],
        batch_size: int,
        num_workers: int,
        base_augmentations: list[Callable[[torch.Tensor], torch.Tensor]],
        train_augmentations: list[Callable[[torch.Tensor], torch.Tensor]],
    ):
        super().__init__()  # make sure LightningDataModule is initialized

        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_augmentations = base_augmentations
        self.train_augmentations = train_augmentations

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=WeatherGenDataset(
                files=self.train_files,
                augmentations=Compose(self.base_augmentations + self.train_augmentations),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=WeatherGenDataset(
                files=self.val_files,
                augmentations=Compose(self.base_augmentations),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def load_all_images() -> list[Path]:
    all_files = []
    for radar in RADARS:
        all_files += [x[1] for x in datetime_glob.walk(PATH_FORMAT.format(radar=radar))]
    return all_files
