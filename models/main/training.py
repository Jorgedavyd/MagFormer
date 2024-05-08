from lightning.pytorch.cli import LightningCLI

from .transformer import MagFormer
from ...data import FullDataModule

## Run

if __name__ == "__main__":

    cli = LightningCLI(
        model_class=MagFormer,
        datamodule_class=FullDataModule,
        seed_everything_default=42,
        trainer_defaults={
            "max_epochs": 200,
            "accelerator": "gpu",
        },
    )

    model = MagFormer(
        {
            "encoder_lr": cli.model.encoder_lr,
            "encoder_wd": cli.model.encoder_wd,
            "decoder_lr": cli.model.decoder_lr,
            "decoder_wd": cli.model.decoder_wd,
            "optimizer": cli.model.optimizer,
            "layers": cli.model.layers,
            "alpha": cli.model.alpha,
            "lambda": cli.model.lambd,
            "x_d_model": cli.model.x_d_model,
            "X_d_model": cli.model.X_d_model,
            "x_seq_len": cli.model.x_seq_len,
            "X_seq_len": cli.model.X_seq_len,
            "architecture": cli.model.architecture,
            "n_heads": cli.model.n_heads,
            "degree": cli.model.degree
        }
    )
