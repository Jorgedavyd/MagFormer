from torch import Tensor, nn
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Union
import torch
from utils import interval_time
from numpy.typing import NDArray

## path utils
dtype: str = '%y/%m/%d'

def soho_path(date_: datetime) -> str:
    return f"./data/SOHO/{date_.strftime(dtype)}.csv"

def wind_path(date_: datetime) -> str:
    return f"./data/WIND/{date_.strftime(dtype)}.csv"

def new_path(date_: datetime) -> str:
    return f"./data/DSCOVR/{date_.strftime(dtype)}.csv"

## Dataframe utils
def create_df(input: Tensor, scrap_date: List[datetime], delta_t: timedelta) -> pd.DataFrame:
    array: NDArray = input.detach().cpu().squeeze(0).numpy()
    start_date, end_date = scrap_date
    time_index = pd.date_range(start=start_date, end=end_date, freq=delta_t)
    df = pd.DataFrame(data=array, index=time_index)
    return df

def save_data(input: Tensor, list_date: List[datetime], delta_t: timedelta) -> None:
    for date_ in list_date:
        create_df(input, date_, delta_t).to_csv(new_path(date_))

def model_pipeline(scrap_date_list: List[Tuple[datetime, datetime]], model_path = './synthetic/model.pt', delta_t: timedelta = timedelta(minutes = 5)) -> None:
    model = torch.jit.load(model_path)
    for scrap_date_tuple in scrap_date_list:
        scrap_date: List[datetime] = interval_time(scrap_date_tuple)
        input_data: Tensor = get_data(scrap_date)
        out: Tensor = model(input_data)
        save_data(out, scrap_date, delta_t)

def get_data(scrap_date: List[datetime]) -> Tensor:
    main_df = pd.DataFrame()
    for date_ in scrap_date:

        # Get data from WIND and SOHO
        soho_df = pd.read_csv(soho_path(date_))
        wind_df = pd.read_csv(wind_path(date_))

        df = pd.concat([soho_df, wind_df], axis = 1)

        main_df = pd.concat([main_df, df], axis = 0)

    return torch.from_numpy(main_df.values).unsqueeze(0)

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



