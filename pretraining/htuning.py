import optuna
from lightorch import htuning
from .model import Model
from .dataset import datamodule

def define_hyp(trial: optuna.trial.Trial):

    encoder_lr = trial.suggest_float("encoder_lr", 1e-8, 1e-2, log=True)
    encoder_wd = trial.suggest_float("encoder_wd", 1e-8, 1e-3, log=True)
    decoder_lr = trial.suggest_float("decoder_lr", 1e-8, 1e-2, log=True)
    decoder_wd = trial.suggest_float("decoder_wd", 1e-8, 1e-3, log=True)
    layers = trial.suggest_int("layers", 1, 2)
    alpha_1 = trial.suggest_float("alpha_1", 0, 1)
    alpha_2 = trial.suggest_float("alpha_2", 0, 1)
    alpha_3 = trial.suggest_float("alpha_3", 0, 1)
    alpha_4 = trial.suggest_float("alpha_4", 0, 1)
    alpha_5 = trial.suggest_float("alpha_5", 0, 1)
    alpha_6 = trial.suggest_float("alpha_6", 0, 1)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rms", "sgd"])

    return {
        "encoder_lr": encoder_lr,
        "encoder_wd": encoder_wd,
        "decoder_lr": decoder_lr,
        "decoder_wd": decoder_wd,
        "optimizer": optimizer,
        "layers": layers,
        "alpha": [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6],
    }

if __name__ == '__main__':
    htuning(
        model_class = Model,
        hparam_objective = define_hyp,
        datamodule = datamodule,
        valid_metrics = 'MSE', # TODO: Definir
        directions = 'minimize',
        precision = 'medium',
        n_trials = 25,
    )
