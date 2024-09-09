from collections.abc import Callable
from typing import List, Tuple
from datetime import datetime, timedelta, time
from glob import glob
import pandas as pd

from synthetic.utils import datetime_interval, timedelta_to_freq
# TODO: Create a way to fill all of the missing dates for DSCOVR and ACE with this script

def get_missing_dates(filepath_callables: List[Callable[[str], str]] | Callable[[str], str] , date_format: str = "%Y%m%d") -> List[Tuple[datetime,datetime]]:
    if isinstance(filepath_callables, (tuple, list)):
        for func in filepath_callables:
            get_missing_dates(func, date_format)
    else:



def fill_missing_dates(
    missing_dates: List[Tuple[datetime, datetime]],
    step_size: int,
    model,
    dscovr_filepath: Callable[[str], str],
    ace_filepath: Callable[[str], str],
    dscovr_columns: List[str],
    ace_columns: List[str]
) -> None:
    new_step_size: timedelta = timedelta(minutes=step_size)
    for scrap_date in missing_dates:
        fill_single(scrap_date,new_step_size, model, dscovr_filepath, ace_filepath, dscovr_columns, ace_columns)

def fill_single(
        scrap_date: Tuple[datetime, datetime],
        step_size: timedelta,
        model,
        dscovr_filepath: Callable[[str], str],
        ace_filepath: Callable[[str], str],
        dscovr_columns: List[str],
        ace_columns: List[str]
) -> None:
    new_scrap_date: List[str] = datetime_interval(*scrap_date, timedelta(days = 1))

    for date in new_scrap_date:
        dscovr_path: str = dscovr_filepath(date)
        ace_path: str = ace_filepath(date)

        start = pd.Timestamp.combine(date, time.min)
        end = pd.Timestamp.combine(date, time.max)

        index = pd.date_range(start = start, end = end, freq = timedelta_to_freq(step_size))

        input = ... ## get the data from the other satelitesc

        out = model(input).squeeze(0).detach().cpu().numpy()

        dscovr_value = out[:5] ## SET THE CORRECT ONE
        ace_value = out[5:]

        dscovr_df: pd.DataFrame = pd.DataFrame(dscovr_value, index = index, columns = dscovr_columns)
        ace_df: pd.DataFrame = pd.DataFrame(ace_value, index = index, columns = ace_columns)

        dscovr_df.to_csv(dscovr_path)
        ace_df.to_csv(ace_path)


if __name__ == '__main__':
    model_path: str = './synthetic/model.pt'
    missing_dates: List[Tuple[datetime, datetime]] = get_missing_dates()
    step_size: int = 5
    model = Model().load(model_path)
    dscovr_filepath: Callable[[str], str] = './data/DSCOVR/L1'
    ace_filepath: Callable[[str], str] = './data/ACE/'
    dscovr_columns: List[str] = DSCOVR().columns
    ace_columns: List[str] = ACE().columns
    dates = get_missing_dates('./data/ACE/')
    fill_missing_dates(
        missing_dates,
        step_size,
        model,
        dscovr_filepath,
        ace_filepath,
        dscovr_columns,
        ace_columns
    )
