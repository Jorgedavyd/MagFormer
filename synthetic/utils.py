from torch import Tensor, nn
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import torch
from utils import interval_time

dtype: str = '%y/%m/%d'

def soho_path(date_: datetime) -> str:
    return f"./data/SOHO/{date_.strftime(dtype)}.csv"

def wind_path(date_: datetime) -> str:
    return f"./data/WIND/{date_.strftime(dtype)}.csv"

def new_path(date_: datetime) -> str:
    return f"./data/DSCOVR/{date_.strftime(dtype)}.csv"

def create_df(input: Tensor, scrap_date: datetime, delta_t: timedelta) -> pd.DataFrame:
    # Create the dataframe

def save_data(input: Tensor, scrap_date: List[datetime], delta_t: timedelta) -> None:
    for date_ in scrap_date:
        df = create_df(input, date_, delta_t)
        df.to_csv(new_path(date_))

def model_pipeline(scrap_date_list: List[Tuple[datetime, datetime]], model_path = './models/main.pt', delta_t: timedelta = timedelta(minutes = 5)) -> None:
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
