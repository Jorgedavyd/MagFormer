from lightorch.nn import SelfAttention, Transformer, TransformerCell, FFN_SwiGLU
from lightorch.training.supervised import Module
from .embedding import JointPositionalEncoding
from torch import nn, Tensor
import torch

class MagFormer(Transformer):
    def __init__(
        self,
        ## setup the parameters
    ) -> None:
        super().__init__(
                positional_encoding = JointPositionalEncoding(),
                encoder = TransformerCell(
                    self_attention=SelfAttention(),
                    ffn=FFN_SwiGLU(),
                    prenorm=nn.RMSNorm()
                ),
                fc = nn.Linear(),
                n_layers = 4,
            )

    def forward(self, vae_recons: Tensor, image_recons: Tensor) -> Tensor:
        return super().forward(torch.cat([vae_recons, image_recons], dim = -2))

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model: MagFormer = MagFormer(**hparams)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
