from starstream import DSCOVR, WIND, SOHO, DataDownloading
import asyncio
# define getting the scrap_date_list
scrap_date_list = ...

if __name__ == '__main__':
    asyncio.run(DataDownloading(
        [DSCOVR(), # Defining the label
         SOHO.CELIAS_SEM(), SOHO.CELIAS_PM(), WIND.MAG(), WIND.TDP_PM()],
        scrap_date_list
    ))
