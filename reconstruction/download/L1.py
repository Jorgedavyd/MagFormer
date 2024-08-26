from date_prep import general_dates, merge_scrap_date_lists
from starstream import DataDownloading, OMNI

if __name__ == '__main__':
    DataDownloading(
        OMNI,
        merge_scrap_date_lists(
            general_dates('post_2016.txt'),
            general_dates('pre_2016.txt')
        )
    )

