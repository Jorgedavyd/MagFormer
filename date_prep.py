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
    assert (os.path.exists(path)), "Not valid model_type or name, path not found"
    with open(path, 'r') as file:
        lines = file.readlines()
        dates = map(lambda x: datetime.strptime(x.split()[1], '%Y/%m/%d'), lines)
        scrap_date_list: List[Tuple[datetime, datetime]] = []
        for date in dates:
            if len(scrap_date_list) != 0:
                match_flag: bool = False
                for idx, scrap_date in enumerate(scrap_date_list):
                    if main_condition(date, scrap_date[0]) or main_condition(date, scrap_date[1]):
                        scrap_date_list[idx] = date_union(date, scrap_date)
                        match_flag = True
                        break
                if not match_flag:
                    scrap_date_list.append((date - timedelta(days = 2), date - timedelta(days = 2)))
            else:
                scrap_date_list.append((date - timedelta(days = 2), date - timedelta(days = 2)))
        return scrap_date_list

def merge_scrap_date_lists(first: List[Tuple[datetime, datetime]], second: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    out_scrap_date_list: List[Tuple[datetime, datetime]] = []
    for second_tuple in second:
        for first_tuple in first:
            if first_tuple[0] < second_tuple[0] < first_tuple[1]:
                if first_tuple[1] < second_tuple[1]:
                    out_scrap_date_list.append((first_tuple[0], second_tuple[1]))
                else:
                    pass
            elif first_tuple[0] < second_tuple[1] < first_tuple[1]:
                if second_tuple[0] < first_tuple[0]:
                    out_scrap_date_list.append((second_tuple[0], first_tuple[1]))
                else:
                    pass
            else:
                out_scrap_date_list.append(second_tuple)
    out_scrap_date_list.extend(first)
    return sorted(out_scrap_date_list, key = lambda key: key[0])
