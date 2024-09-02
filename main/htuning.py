import optuna
from lightorch import htuning
from .transformer import MainBackbone
from .dataset import datamodule

def define_hyp(trial: optuna.trial.Trial):

    lr = trial.suggest_float("decoder_lr", 1e-8, 1e-2, log=True)
    wd = trial.suggest_float("decoder_wd", 1e-8, 1e-3, log=True)
    layers = trial.suggest_int("layers", 1, 2)
    alpha_1 = trial.suggest_float("alpha_1", 0, 1, log = True)
    alpha_2 = trial.suggest_float("alpha_2", 0, 1, log = True)
    alpha_3 = trial.suggest_float("alpha_3", 0, 1, log = True)
    alpha_4 = trial.suggest_float("alpha_4", 0, 1, log = True)
    alpha_5 = trial.suggest_float("alpha_5", 0, 1, log = True)
    alpha_6 = trial.suggest_float("alpha_6", 0, 1, log = True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rms", "sgd"])

    return {
        "lr": lr,
        "wd": wd,
        "optimizer": optimizer,
        "layers": layers,
        "alpha": [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6],
    }

if __name__ == '__main__':
    htuning(
        model_class = MainBackbone,
        hparam_objective = define_hyp,
        datamodule = datamodule,
        valid_metrics = 'MSE', # TODO: Definir
        directions = 'minimize',
        precision = 'medium',
        n_trials = 25,
    )
