from datetime import timedelta, datetime
from date_prep import general_dates
from ..dataset import ACEDataset, DSCOVRDataset, DatasetLevel2, DatasetLevel3, DefaultDataModule, WINDDataset, SOHODataset
from torch import Tensor
import torch
from typing import Tuple, List
"""
Dataset architecture

scrap_date_list -> (DSCOVR, ACE)input, (WIND, SOHO)output

"""

class SyntheticTask(DatasetLevel3):
    def __init__(self, base_scrap_date: List[Tuple[datetime, datetime]], step_size: timedelta) -> None:
        dscovr = DatasetLevel2(DSCOVRDataset, base_scrap_date, step_size)
        ace = DatasetLevel2(ACEDataset, base_scrap_date, step_size)
        wind = DatasetLevel2(WINDDataset, base_scrap_date, step_size)
        soho = DatasetLevel2(SOHODataset, base_scrap_date, step_size)
        super().__init__(dscovr, ace, wind, soho)
    def __getitem__(self, idx) -> Tuple[Tensor, ...]:
        dscovr, ace, wind, soho = super().__getitem__(idx)
        return torch.cat([dscovr, ace], dim = -1), torch.cat([wind, soho], dim = -1)

class DataModule(DefaultDataModule):
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool, train_p: float, step_size: int) -> None:
        base_scrap_date: List[Tuple[datetime, datetime]] = general_dates('post_2016')
        step_size: timedelta = timedelta(minutes = step_size)
        super().__init__(SyntheticTask(base_scrap_date, step_size), batch_size, num_workers, pin_memory, train_p)
