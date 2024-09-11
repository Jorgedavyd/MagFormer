import optuna
from lightorch import htuning
from .model import
from .dataset import datamodule

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
        datamodule = datamodule,
        valid_metrics = 'MSE', # TODO: Definir
        directions = 'minimize',
        precision = 'medium',
        n_trials = 25,
    )
