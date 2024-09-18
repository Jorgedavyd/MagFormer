from typing import List
from ..mhd_constraints import MainCriterion

valid_criterions: List[str] = [
    'GaussLawMagnetismConstraint',
    'GaussLawElectrostaticConstraint',
    'Ohm',
    'ForcingCriterion',
]

factors: List[float] = [
    1.,
    1.,
    1.,
    1.,
]

criterion = MainCriterion(
    valid_criterions,
    factors
)
