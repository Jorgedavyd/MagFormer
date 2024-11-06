from ..MHD.mhd_constraints import MainCriterion
from typing import List

## temp, vel, B, Flow pressure, alfven,
valid_criterions: List[str] = [
    '',
    '',
]

factors: List[float] = [
    1.,
    1.,
]




criterion = MainCriterion(
    valid_criterions,
    factors
)

