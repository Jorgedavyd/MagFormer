from lightorch.nn import DnPositionalEncoding, RotaryPositionalEncoding
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from datetime import timedelta
# Defining main


# Joining the Derivative based positional encoding
class JointPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, step_size: timedelta, degree: int, edge_order):
        super().__init__()
        self.dn_pos = DnPositionalEncoding(step_size, degree, edge_order)
        self.rot_pe = RotaryPositionalEncoding(d_model, seq_len)
        def forward(self, x_n):
            out = self.dn_pos(x_n)
            out = self.rot_pe(out)
            return out

# Patch embbedding and rotary position embeddings
class PatchEmbed_2DCNN(nn.Module):
    def __init__(
        self,
        d_model: int,
        pe,
        feature_extractor,
        architecture: tuple,
        hidden_activations: tuple,
        dropout: float = 0.1,
        ):
        super().__init__()
        """
        B: batch size
        F: Frames
        C: Channels
        H_div: number of vertical cuts
        W_div: number of horizontal cuts
        H_div*W_div: total patches
        h: H/H_div
        w: W/W_div
        """
        self.d_model = d_model
        self.feature_extractor  = feature_extractor(d_model, architecture, hidden_activations, dropout)
        self.pe = pe
        def forward(
            self,
            X: torch.Tensor
            ):
            B,F,C,H,W = X.shape
            # -> (B*F,C,H, W)
            X = X.view(B*F, C, H, W)
            # -> (B,F, embed_size)
            out = self.feature_extractor(X).view(B, F, -1)
            # -> (B,F, embed_size)
            X = self.pe(out)
            return X

# Cross Encoder block to train both modules at the same time
class CrossEncoderBlock(nn.Module):
    def __init__(
        self,
        x_d_model: int,
        X_d_model: int,
        x_seq_len: int,
        X_seq_len: int,
        n_heads: int,
        dropout: float = 0.

        ) -> None:
        super().__init__()

        # Tabular data
        self.res_0 = nn.ModuleList([PreNormResidualConnection(RootMeanSquaredNormalization(x_d_model), dropout) for _ in range(3)])
        self.self_att_0 = ModedTimeSeriesAttention(x_d_model, x_seq_len, n_heads, x_d_model, x_d_model)
        self.cross_att_0 = ModedTimeSeriesAttention(X_d_model, X_seq_len, n_heads, X_d_model, x_d_model)
        x_poswise_mlp = PosWiseFFN(x_d_model, x_d_model)
        x_seqwise_mlp = SeqWiseMLP(x_seq_len, x_d_model)
        self.ffn_0 = lambda x_n: x_poswise_mlp(x_n) + x_seqwise_mlp(x_n)

        # Imagery data
        self.res_1 = nn.ModuleList([PreNormResidualConnection(RootMeanSquaredNormalization(x_d_model), dropout) for _ in range(3)])
        self.self_att_1 = ModedTimeSeriesAttention(X_d_model, X_seq_len, n_heads, X_d_model, X_d_model)
        self.cross_att_1 = ModedTimeSeriesAttention(x_d_model, x_d_model, n_heads, x_d_model, X_d_model)
        X_poswise_mlp = PosWiseFFN(X_d_model, X_d_model)
        X_seqwise_mlp = SeqWiseMLP(X_seq_len, X_d_model)
        self.ffn_1 = lambda x_n: X_poswise_mlp(x_n) + X_seqwise_mlp(x_n)

        def forward(self, x: Tensor, X: Tensor) -> Tensor:
            # Self attention through the tabular
            x = self.res_0[0](x,lambda x: self.self_att_0(x))
            # Self attention through the imagery
            X = self.res_1[0](X, lambda X: self.self_att_1(X))

            # Cross attention through the tabular
            out = self.res_0[1]((x, X),lambda x, X: self.cross_att_0(x, X))
            # Cross attention through the imagery
            OUT = self.res_1[1](x,lambda x: self.cross_att_1(x))

            # FFN through the tabular
            out = self.res_0[2](out, lambda x: self.ffn_0(x))
            # FFN through the imagery
            OUT = self.res_1[2](OUT, lambda x: self.ffn_1(x))

            return out, OUT


# MagFormer backbone
class MagFormer(LightningModule):
    def __init__(
        self,
        step_size: timedelta,
        pred_steps: int,
        hparams: dict
        ) -> None:
        super().__init__()

        for k,v in hparams.items():
            setattr(self, k, v)

            self.save_hyperparameters()
            #: B, S, I -> B,S,x_d_model
            self.embed_x = DnPositionalEncoding(step_size, self.degree)

            # B, 1, 256, 256 -> B, S, X_d_model
            self.embed_X = ImageRotaryPositionalEncoding()

            # {(B,X_seq_len,X_d_model), (B,x_seq_len, x_d_model)} -> {(B,X_seq_len,X_d_model), (B,x_seq_len, x_d_model)}
            self.encoder_blocks = nn.ModuleList([
                CrossEncoderBlock(
                    self.x_d_model, self.X_d_model, self.x_seq_len, self.X_seq_len, self.n_heads, self.dropout
                ) for _ in range(self.layers)
            ])
            # B, X
            self.fc_attention = ModedTimeSeriesAttention(self.X_d_model, self.X_seq_len, self.num_heads, self.X_seq_len, self.x_seq_len)
            self.fc = MonteCarloFC(DeepNeuralNetwork(self.x_d_model, pred_steps, self.architecture), self.mc_dropout)

        def forward(self, x: Tensor, X: Tensor) -> Tensor:
            # Through the embedding step
            X = self.embed_X(X)
            x = self.embed_x(x)
            # Through the transformer
            for encoder in self.encoder_blocks:
                x, X = encoder(x,X)
                # Through the monte carlo sampling step
                x = self.fc_attention(x, X)
                return self.fc(x)

        def training_step(self, batch: Tensor) -> Tensor:

        def validation_step(self, batch:Tensor) -> None:

