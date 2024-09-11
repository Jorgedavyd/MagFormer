from date_prep import general_dates, merge_scrap_date_lists
from starstream import DataDownloading, OMNI
from datetime import datetime
from typing import Tuple, List

if __name__ == '__main__':
    scrap_date_list: List[Tuple[datetime, datetime]] = merge_scrap_date_lists(
        general_dates('post_2016.txt'),
        general_dates('pre_2016.txt')
    )

    DataDownloading(
        OMNI(),
        scrap_date_list
    )


