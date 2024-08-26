from starstream import SDO, DataDownloading
from date_prep import general_dates

if __name__ == '__main__':
    DataDownloading(
        SDO.AIA_HR(),
        general_dates('post_2010')
    )
