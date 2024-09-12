from torch import Tensor
from typing import Dict

def batch_data(out: Tensor) -> Dict[str, Tensor]:
    return dict(
        B = out[0:1],
    )
