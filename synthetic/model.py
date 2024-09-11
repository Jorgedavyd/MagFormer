from lightorch.nn.sequential.residual import LSTM, GRU
from lightorch.training.supervised import Module
from torch import nn, Tensor
from typing import Dict, Tuple
import torch
input_size: int = 6
out_size: int = 4

class SyntheticDataModelArchitecture(nn.Module):
    def __init__(self, model_type: str, input_size: int, hidden_size: int, rnn_layers: int, res_layers: int, bias: bool = True, batch_first: bool = True, dropout: float = 0, bidirectional: bool = False, proj_size: int = 0, device: Any | None = None, dtype: Any | None = None) -> None:
        assert (model_type == 'GRU' or model_type == 'LSTM'), "Not valid architecture"
        self.model: LSTM | GRU = LSTM(
            input_size,
            hidden_size,
            rnn_layers,
            res_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype
        ) if model_type == 'LSTM' else GRU(
            input_size,
            hidden_size,
            rnn_layers,
            res_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            device,
            dtype
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.model(x))

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = SyntheticDataModelArchitecture(**hparams)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss_forward(self, batch:Tensor, idx: int) -> Dict[str, Tensor | float]:
        out: Tuple[Tensor, ...] = self.model(batch[0])
        return dict(
            label = torch.stack(out, dim = -1),
            target = batch[1],

            ## setup B,, and all of that stuff
        )
