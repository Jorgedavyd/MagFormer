from corkit import downloader
from .dates import scrap_date_list

if __name__ == '__main__':
    down = downloader('c2', './data/SOHO/LASCO')
    down(scrap_date_list)
