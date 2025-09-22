from torch import Tensor
from dataclasses import dataclass
from torch.utils.data import Dataset

from torchvision.datasets import FashionMNIST
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToTensor, Identity


@dataclass(kw_only=True)
class FashionMNISTDataset(Dataset):
    base_path: str
    train: bool

    def __post_init__(self) -> None:
        self.ds = FashionMNIST(
            root=self.base_path,
            train=self.train,
            download=True,
            transform=Compose([
                ToTensor(),
                RandomHorizontalFlip() if self.train else Identity()
            ])
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Tensor:
        return self.ds[index][0]
    
    @staticmethod
    def get_train_dataset(base_path: str) -> 'FashionMNISTDataset':
        return FashionMNISTDataset(
            base_path=base_path,
            train=True,
        )
    
    @staticmethod
    def get_val_dataset(base_path: str) -> 'FashionMNISTDataset':
        return FashionMNISTDataset(
            base_path=base_path,
            train=False,
        )
