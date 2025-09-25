import lightning as L
from torch.utils.data import Dataset, DataLoader


class VAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        **kwargs
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            shuffle=True,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ds,
            shuffle=False,
            **self.kwargs,
        )
