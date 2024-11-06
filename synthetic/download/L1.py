from date_prep import general_dates, merge_scrap_date_lists
from starstream import WIND, SOHO, ACE, DSCOVR, DataDownloading
from typing import Tuple, List
from datetime import datetime
import os.path as osp

if __name__ == '__main__':
    scrap_date_list: List[Tuple[datetime, datetime]] = merge_scrap_date_lists(
            general_dates('post_2016'),
            general_dates('pre_2016')
    )
    path: str = '/data/MagFormer'

    DataDownloading(
        [
            ACE.MAG(download_path = osp.join(path, "ACE/MAG")),
            ACE.SWEPAM(download_path = osp.join(path, "ACE/SWEPAM")),
            ACE.EPAM(download_path = osp.join(path, "ACE/EPAM")),
            ACE.SIS(download_path = osp.join(path, "ACE/SIS")),
            SOHO.CELIAS_SEM(download_path = osp.join(path, "SOHO/CELIAS_SEM")),
            SOHO.CELIAS_PM(download_path = osp.join(path, "SOHO/CELIAS_PM")),
            WIND.MAG(root = osp.join(path, "WIND/MAG")),
            WIND.TDP_PM(root = osp.join(path, "WIND/TDP_PM")),
            # DSCOVR.FaradayCup(root = osp.join(path, "DSCOVR/FaradayCup")),
            # DSCOVR.Magnetometer(root = osp.join(path, "DSCOVR/Magnetometer"))
        ],
        scrap_date_list
    )
