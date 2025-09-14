import numpy as np
import torch
import datetime_glob
import os
from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor, RandomVerticalFlip
from lightning import LightningDataModule

TRAIN_RADARS = [
    'KDMX',
    'KUEX'
]
VAL_RADARS = [
    'KGLD'
]
PATH_FORMAT = '/mnt/remote/mp_data_cmuchalek/nexrad/experiment-cache-updated-rhohv-pp-velocity/{radar}/%Y/%m/%d/sample_%Y-%m-%d_%H-%M-%S.npz'
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
        try:
            data = np.load(self.files[index], allow_pickle=True)['arr_0'][()]['input']['DBZH_pseudocappi'][0, 0]
        except Exception as e:
            print(f'Failed to open {self.files[index]}, got error: {e}')
            raise
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


def _filter_files(files: list[Path], cache_file_path: str) -> list[Path]:
    to_skip_file = Path(cache_file_path)
    to_skip = []
    if to_skip_file.exists():
        with open(to_skip_file, 'r') as f:
            for l in f.readlines():
                to_skip.append(l.strip())
    else:
        for idx in tqdm(range(len(files))):
            if os.path.getsize(files[idx]) == 0:
                to_skip.append(files[idx])
                continue
            try:
                _ = np.load(files[idx], allow_pickle=True)
            except Exception as e:
                to_skip.append(files[idx])
        with open(to_skip_file, 'w+') as f:
            for t in to_skip:
                f.write(t + '\n')
    return [x for x in files if x not in to_skip]



def get_train_files() -> list[Path]:
    train_files = []
    for radar in TRAIN_RADARS:
        train_files += [x[1].as_posix() for x in datetime_glob.walk(PATH_FORMAT.format(radar=radar))]
    return _filter_files(train_files, 'to-skip-train.txt')


def get_val_files() -> list[Path]:
    val_files = []
    for radar in VAL_RADARS:
        val_files += [x[1].as_posix() for x in datetime_glob.walk(PATH_FORMAT.format(radar=radar))]
    return _filter_files(val_files, 'to-skip-val.txt')
