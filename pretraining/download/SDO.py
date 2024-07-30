from starstream import SDO
from .dates import scrap_date_list

if __name__ == '__main__':
    DataDownloader(
        SDO.AIA_HR(),
        scrap_date_list
    )
