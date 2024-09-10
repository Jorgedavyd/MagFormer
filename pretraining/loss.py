from typing import List
from ..mhd_constraints import MainCriterion

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

