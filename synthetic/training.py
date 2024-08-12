from lightorch.training.cli import trainer
from model import Model1, Model2

if __name__ == '__main__':
    trainer(
        precision='high',
        deterministic = True,
        seed = 123
    )
