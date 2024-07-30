from lightorch.training.cli import trainer
from ...data import FullDataModule

if __name__ == '__main__':
    trainer(
        FullDataModule,
        deterministic=True,
        seed=123
    )
