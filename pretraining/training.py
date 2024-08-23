from lightorch.training.cli import trainer
from .model import Model

if __name__ == '__main__':
    trainer(
        seed = 123,
        precision = 'high',
        deterministic = True,
    )

