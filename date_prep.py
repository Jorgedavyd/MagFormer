from collections.abc import Callable
from typing import Tuple, List
from datetime import datetime, timedelta
import os

main_condition: Callable[[datetime, datetime], bool] = lambda date, scrap_date: timedelta(days = -4) < scrap_date - date < timedelta(days = 4)

def date_union(first_date: datetime, second_scrap_date: Tuple[datetime, datetime]) -> Tuple[datetime, datetime]:
    if first_date < second_scrap_date[0]:
        return first_date - timedelta(days = 4), second_scrap_date[1]
    elif first_date < second_scrap_date[1]:
        return second_scrap_date
    else:
        return second_scrap_date[0], first_date + timedelta(days = 2)

def general_dates(name: str) -> List[Tuple[datetime, datetime]]:
    path: str = f'./{name}.txt'
    assert os.path.exists(path), "Not valid model_type or name, path not found"

    with open(path, 'r') as file:
        dates = list(map(lambda x: datetime.strptime(x.split()[1], '%Y/%m/%d'), file.readlines()))

    scrap_date_list: List[Tuple[datetime, datetime]] = [(datetime(1990, 10, 10), datetime(1990, 10, 11))]

    for date in dates:
        flag: bool = True
        i = 0
        while i < len(scrap_date_list):
            scrap_date = scrap_date_list[i]
            if main_condition(date, scrap_date[0]) or main_condition(date, scrap_date[1]):
                scrap_date_list[i] = date_union(date, scrap_date)
                flag = False
                break
            i += 1

        if flag:
            scrap_date_list.append((date - timedelta(days=4), date + timedelta(days=2)))

    return scrap_date_list[1:]

def merge_scrap_date_lists(first: List[Tuple[datetime, datetime]], second: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    a: List[Tuple[datetime, datetime]] = sorted(first + second, key=lambda x: x[0])
    out_scrap_date_list: List[Tuple[datetime, datetime]] = []
    i: int = 0
    while i < len(a):
        start, end = a[i]
        while i + 1 < len(a) and a[i + 1][0] <= end:
            end = max(end, a[i + 1][1])
            i += 1
        out_scrap_date_list.append((start, end))
        i += 1

    return out_scrap_date_list

