from date_prep import general_dates, merge_scrap_date_lists
from starstream.utils import DataDownloading
from typing import Tuple, List
from datetime import datetime
from starstream import OMNI

if __name__ == '__main__':
    scrap_date_list: List[Tuple[datetime, datetime]] = merge_scrap_date_lists(
            general_dates('post_2016'),
            general_dates('pre_2016')
    )

    DataDownloading(
        OMNI(),
        scrap_date_list
    )


