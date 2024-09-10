from synthetic.utils import datetime_interval, get_data, get_data_paths, timedelta_to_freq
from datetime import datetime, timedelta, time
from starstream import SOHO, WIND, ACE, DSCOVR
from collections.abc import Callable
from typing import List, Tuple, Set
from torch import Tensor
from .model import Model
import os.path as osp
import pandas as pd
import glob

def get_column_value() -> List[List[str]]:
    return [
        SOHO.CELIAS_PM().variables,
        SOHO.CELIAS_SEM().variables,
        WIND.MAG().variables,
        WIND.TDP_PM().variables,
        ACE.SWEPAM().variables,
        ACE.EPAM().variables,
        ACE.SIS().variables,
        ACE.MAG().variables,
        DSCOVR().fc_var,
        DSCOVR().mg_var,
    ]

def segment_output(out: Tensor) -> List[Tensor]:
    column_names: List = get_column_value().insert(0, [])
    idxs: List[slice] = list(map(lambda x, y: slice(x, y), zip(column_names[:-1], column_names[1:])))
    return [out[idx] for idx in idxs]

def get_missing_paths(interval_time: Tuple[datetime, datetime], filepath_callables: List[Callable[[str], str]] | Callable[[str], str]) -> List[List[str]] | List[str]:
    if isinstance(filepath_callables, (tuple, list)):
        return [get_missing_paths(interval_time, func) for func in filepath_callables]
    else:
        new_scrap_date: List[str] = datetime_interval(*interval_time, timedelta(days = 1))
        csv_paths: Set[str] = set(glob.glob(filepath_callables('*')))
        expected_paths: Set[str] = set([filepath_callables(date) for date in new_scrap_date])
        return list(expected_paths - csv_paths)

def fill_missing_dates(
    missing_paths: List[List[str]],
    column_values: List[List[str]],
    step_size: int,
    model,
) -> None:
    new_step_size: timedelta = timedelta(minutes=step_size)
    for scrap_path in missing_paths:
        fill_single(scrap_path, column_values, new_step_size, model)

def fill_single(
        scrap_path: List[str],
        column_values: List[List[str]],
        step_size: timedelta,
        model,
) -> None:
    assert (len(scrap_path) == len(column_values)), "Not valid scrap_path nor columns_values"

    input_sample_paths: List[str] = scrap_path[:4]
    output_sample_paths: List[str] = scrap_path[4:]

    date = datetime.strptime(osp.basename(scrap_path[0])[:-4], '%Y%m%d')

    start = pd.Timestamp.combine(date, time.min)
    end = pd.Timestamp.combine(date, time.max)

    index = pd.date_range(start = start, end = end, freq = timedelta_to_freq(step_size))

    input = get_data(input_sample_paths) ## the other input that are not ace and dscovr

    out = model(input).squeeze(0).detach().cpu().numpy()

    segmented_out: List[Tensor] = segment_output(out)

    for path, out, columns in zip(output_sample_paths, segmented_out, column_values):
        pd.DataFrame(out, index = index, columns = columns).to_csv(path)

if __name__ == '__main__':
    model_path: str = './synthetic/model.pt'
    interval_time: Tuple[datetime, datetime] = (datetime(2000, 1, 1), datetime(2024, 8, 29))
    step_size: int = 5

    missing_paths: List[List[str]] = get_missing_paths(
        interval_time,
        get_data_paths()
    )

    column_values: List[List[str]] = get_column_value()

    model = Model().load_state_dict(model_path)

    fill_missing_dates(
        missing_paths,
        column_values,
        step_size,
        model,
    )
