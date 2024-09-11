from lightorch.htuning import htuning
from .data import DataModule
from .loss import criterion
from .model import Model
from typing import List
import optuna

labels : List[str] | str = criterion().labels

def objective(trial: optuna.trial.Trial) -> Dict[str, float|int|str|List[str]|Dict[str, float|int]]:
    return dict(
        optimizer = 'adam',
        scheduler = 'onecycle',
        triggers = [''],
        optimizer_kwargs = dict(),
        scheduler_kwargs = dict(),
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log = True),
        weight_decay= trial.suggest_float("weight_decay", 1e-6, 1e-2, log = True),
        layers = trial.suggest_int("layers", 1, 5),
        hidden_size = trial.suggest_int("layers", 10, 100),
    )

if __name__ == '__main__':
    htuning(
            model_class = Model,
            hparam_objective = objective,
            datamodule = DataModule,
            valid_metrics = labels,
            datamodule_kwargs = dict(
                pin_memory=True, num_workers=8, batch_size=32
            ),
            directions = ['minimize' for _ in range(len(labels))],
            precision = 'high',
            n_trials = 150,
            trainer_kwargs = dict(
                logger=True,
                enable_checkpointing=False,
                max_epochs=10,
                accelerator="cuda",
                devices=1,
                log_every_n_steps=22,
                precision="bf16-mixed",
                limit_train_batches=1 / 3,
                limit_val_batches=1 / 3,
    )
    )

