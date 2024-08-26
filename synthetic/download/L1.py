from datetime import datetime
from starstream import DSCOVR, WIND, SOHO
from starstream.utils import DataDownloading
from date_prep import general_dates, merge_scrap_date_lists
import asyncio
from typing import Tuple, List

if __name__ == '__main__':
    scrap_date_list: List[Tuple[datetime, datetime]] = merge_scrap_date_lists(
            general_dates('post_2016'),
            general_dates('pre_2016')
    )
    asyncio.run(DataDownloading(
        [DSCOVR(), # Defining the label
         SOHO.CELIAS_SEM(), SOHO.CELIAS_PM(),
         WIND.MAG(), WIND.TDP_PM()],
        scrap_date_list
    ))
