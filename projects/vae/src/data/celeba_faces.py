import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize
from sklearn.model_selection import train_test_split

BASE_PATH = '/data/datasets/celeba/images'
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
    
    @staticmethod
    def get_train_dataset(base_path: str, train_size: float) -> 'FacesDataset':
        all_images = list(Path(base_path).iterdir())
        wanted_images = train_test_split(all_images, train_size=train_size)[0]
        return FacesDataset(
            images=wanted_images,
            augmentations=Compose(BASE_AUGMENTATIONS + TRAIN_AUGMENTATIONS)
        )
    
    @staticmethod
    def get_val_dataset(base_path: str, train_size: float) -> 'FacesDataset':
        all_images = list(Path(base_path).iterdir())
        wanted_images = train_test_split(all_images, train_size=train_size)[1]
        return FacesDataset(
            images=wanted_images,
            augmentations=Compose(BASE_AUGMENTATIONS)
        )
