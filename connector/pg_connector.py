import pandas as pd
from conf.conf import logging

def get_data(link: str) -> pd.DataFrame:

    logging.info("extracting df")
    df = pd.read_csv(link)
    logging.info("DF is extracted")
    return df