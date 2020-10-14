import csv
import sys
import pandas as pd
import numpy
import tempfile
import numpy as np


def preprocess_calendar():
    listing_data = df_calendar['listing_id'].astype(dtype='int32')
    listing_data = listing_data.astype(dtype='int64')
    if not listing_data.equals(df_calendar['listing_id']):
        print("ERROR: casting failed. ")
    else:
        df_calendar['listing_id'] = df_calendar['listing_id'].astype(dtype='int32')

    df_calendar['available'].replace(to_replace='t', value=True, inplace=True)
    df_calendar['available'].replace(to_replace='f', value=False, inplace=True)
    df_calendar['available'] = df_calendar['available'].astype(dtype='bool')

    df_calendar.loc[:, 'price'] = df_calendar.loc[:, 'price'].str.strip('$')
    df_calendar['price'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_calendar['price'] = pd.to_numeric(df_calendar['price'], 'raise', 'signed')
    df_calendar['price'] = df_calendar['price'].fillna(value=-1)
    price_data = df_calendar['price'].astype('int32')
    price_data = price_data.astype('float64')

    if not price_data.equals(df_calendar['price']):
        print("ERROR: casting failed. ")
    else:
        df_calendar['price'] = df_calendar['price'].astype(dtype='int32')

    pd.to_datetime(df_calendar.loc[:, 'date'], infer_datetime_format=True)
    # df_calendar.info()

def preprocess_listings():
    df_listings.info()




with open("data/calendar.csv") as f:
    df_calendar = pd.read_csv("data/calendar.csv")
with open("data/listings.csv") as f:
    df_listings = pd.read_csv("data/listings.csv")
with open("data/reviews.csv") as f:
    df_reviews = pd.read_csv("data/reviews.csv")

preprocess_listings()
#preprocess_calendar()
