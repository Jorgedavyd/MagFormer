from lightorch.nn import DnPositionalEncoding, RotaryPositionalEncoding
from torch import nn
from datetime import timedelta

class JointPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, step_size: timedelta, degree: int, edge_order):
        super().__init__()
        self.dn_pos = DnPositionalEncoding(step_size, degree, edge_order)
        self.rot_pe = RotaryPositionalEncoding(d_model, seq_len)
    def forward(self, x_n):
        out = self.dn_pos(x_n)
        out = self.rot_pe(out)
        return out
