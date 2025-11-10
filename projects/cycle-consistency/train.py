import yaml
import sys
from lightning.pytorch import seed_everything

from gorq_ml.training import train

from src.data.moving_mnist import MovingMnistCycleConsistencyDataset
from src.models.cycle_consistency_pl import CycleConsistencyPL



def main() -> None:
    seed_everything(42)
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    train_ds, val_ds = MovingMnistCycleConsistencyDataset.get_train_val_dataset(**config['dataset_kwargs'])
    train(
        config=config,
        l_module=CycleConsistencyPL,
        train_ds=train_ds,
        val_ds=val_ds,
    )

if __name__ == '__main__':
    main()
