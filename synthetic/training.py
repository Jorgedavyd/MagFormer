from lightorch.training.cli import trainer

if __name__ == '__main__':
    trainer(
        precision='high',
        deterministic = True,
        seed = 123
    )
