from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from collections.abc import Callable
from torch import Tensor
from typing import List, Union, Dict
import pandas as pd
import torch

def get_data_paths() -> List[Callable[[str], str]]:
    return [
        lambda date: f'./data/SOHO/CELIAS_PM/{date}.csv',
        lambda date: f'./data/SOHO/CELIAS_SEM/{date}.csv',
        lambda date: f'./data/WIND/MAG/{date}.csv',
        lambda date: f'./data/WIND/TDP_PM/{date}.csv',
        lambda date: f'./data/ACE/SWEPAM/{date}.csv',
        lambda date: f'./data/ACE/EPAM/{date}.csv',
        lambda date: f'./data/ACE/SIS/{date}.csv',
        lambda date: f'./data/ACE/MAG/{date}.csv',
        lambda date: f'./data/DSCOVR/L2/faraday/{date}.csv',
        lambda date: f'./data/DSCOVR/L2/magnetometer/{date}.csv',
    ]

def get_data(paths: List[str]) -> Tensor:
    df_list: List[pd.DataFrame] = []
    for path in paths:
        df_list.append(pd.read_csv(path))
    return torch.from_numpy(pd.concat(df_list, axis = 1).values).unsqueeze(0)

def timedelta_to_freq(timedelta_obj) -> str:
    total_seconds = timedelta_obj.total_seconds()

    if total_seconds % 1 != 0:
        raise ValueError("Timedelta must represent a whole number of seconds")

    days = total_seconds // (24 * 3600)
    hours = (total_seconds % (24 * 3600)) // 3600
    minutes = ((total_seconds % (24 * 3600)) % 3600) // 60
    seconds = ((total_seconds % (24 * 3600)) % 3600) % 60

    freq_str = ""

    if days > 0:
        freq_str += str(int(days)) + "day"
    if hours > 0:
        freq_str += str(int(hours)) + "hour"
    if minutes > 0:
        freq_str += str(int(minutes)) + "min"
    if seconds > 0:
        freq_str += str(int(seconds)) + "sec"

    return freq_str


def datetime_interval(
    init: datetime,
    last: datetime,
    step_size: Union[relativedelta, timedelta],
    output_format: str = "%Y%m%d",
) -> List[str]:
    current_date = init
    date_list = []
    while current_date <= last:
        date_list.append(current_date.strftime(output_format))
        current_date += step_size
    return date_list
