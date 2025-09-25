import os
import yaml
import lightning as L
from clearml import Task
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from argparse import ArgumentParser
from lightning.pytorch import callbacks
from lightning.pytorch import seed_everything


from src import data
from src.data.data_module import VAEDataModule
from src.pl.vae import VAELightning


parser = ArgumentParser(description='Train a WeatherGen model')
parser.add_argument(
    '--config',
    help='Path to the config file.',
    required=True
)
parser.add_argument(
    '--name',
    required=True,
    type=str,
    help='Base name of the experiments to run.'
)

def train(
        config: dict,
) -> None:
    dataset = getattr(data, config['data']['dataset_name'])
    train_ds = dataset.get_train_dataset(
        **config['data'].get('common_kwargs', {}),
        **config['data'].get('train_kwargs', {}),
    )
    val_ds = dataset.get_val_dataset(
        **config['data'].get('common_kwargs', {}),
        **config['data'].get('val_kwargs', {}),
    )

    dl = VAEDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        **config['dataloader'],
    )
    logger.info(f'Train samples: {len(dl.train_dataloader().dataset)}') # type: ignore
    logger.info(f'Val samples: {len(dl.val_dataloader().dataset)}') # type: ignore
    model = VAELightning(
        **config['model'],
        val_dataset=dl.val_dataloader().dataset,
        train_dataset=dl.train_dataloader().dataset,
    )

    config['checkpoint']['dirpath'] = os.path.join(config['checkpoint']['dirpath'], config['task']['task_name'])
    task: Task = Task.init(**config['task'])
    task.connect(config)
    trainer = L.Trainer(
        **config['trainer'],
        callbacks=[callbacks.ModelCheckpoint(**config['checkpoint'])] + [
            getattr(callbacks, k)(**v) for k, v in config.get('callbacks').items() # type: ignore
        ], # type: ignore
        deterministic=True,
    )
    trainer.fit(model, dl, ckpt_path=config.get('ckpt_path'))

def main() -> None:
    seed_everything(42)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['task']['task_name'] = args.name
    train(config)

if __name__ == '__main__':
    main()
