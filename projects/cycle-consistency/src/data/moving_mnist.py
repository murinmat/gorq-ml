import torch
from typing import Tuple
from torch import Tensor
from torchvision.datasets import MovingMNIST


class MovingMnistCycleConsistencyDataset(MovingMNIST):
    def __init__(self, *args, mnist_sample_indices: list[int], num_interp: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.mnist_sample_indices = mnist_sample_indices
        self.num_interp = num_interp

    def __len__(self) -> int:
        return 18 * len(self.mnist_sample_indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        samples_per_idx = 18
        sample_idx = index // samples_per_idx
        sample_subidx = index - samples_per_idx * sample_idx
        data = super().__getitem__(sample_idx)
        return {
            'im1': data[sample_subidx] / 255.,
            'im2': data[sample_subidx+1] / 255.,
            'im3': data[sample_subidx+2] / 255.,
            't': torch.randint(0, self.num_interp, size=(1,)),
        }

    @staticmethod
    def get_train_val_dataset(
        data_root: str,
        train_ratio: float,
        num_interp: int
    ) -> Tuple['MovingMnistCycleConsistencyDataset', 'MovingMnistCycleConsistencyDataset']:
        indices_list = list(range(10_000))
        cutoff = int(len(indices_list) * train_ratio)
        train = MovingMnistCycleConsistencyDataset(
            root=data_root,
            download=True,
            mnist_sample_indices=indices_list[:cutoff],
            num_interp=num_interp
        )
        val = MovingMnistCycleConsistencyDataset(
            root=data_root,
            download=True,
            mnist_sample_indices=indices_list[cutoff:],
            num_interp=num_interp
        )
        return train, val
