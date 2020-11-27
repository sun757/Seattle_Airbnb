import csv
import sys
import pandas as pd
import numpy
import tempfile
import numpy as np
import string

def preprocess_calendar():
    df_calendar['listing_id'] = df_calendar['listing_id'].astype(dtype='Int32')

    df_calendar.loc[:, 'price'] = df_calendar.loc[:, 'price'].str.strip('$')
    df_calendar['price'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_calendar['price'] = pd.to_numeric(df_calendar['price'], 'raise', 'signed')
    df_calendar['price'] = df_calendar['price'].astype(dtype='Int32')

    def split_date(column):
        mylist = column.split("-")
        month = mylist[1]
        return month

    df_calendar['month'] = df_calendar.date.apply(split_date)

    # df_calendar.info()


def preprocess_listings():
    for j in df_listings.columns:
        df_listings[j]=df_listings[j].replace(to_replace={r'[^ -~]'},value='',regex=True)

    df_listings.drop(1939, 0,inplace=True)

    # delete cols
    df_listings.drop('scrape_id', 1,inplace=True)
    df_listings.drop('last_scraped', 1,inplace=True)
    df_listings.drop('experiences_offered', 1,inplace=True)
    df_listings['host_response_time'].replace(to_replace='N/A', value=pd.NA, inplace=True)
    df_listings['host_response_rate'].replace(to_replace='N/A', value=pd.NA, inplace=True)
    df_listings.drop('host_acceptance_rate', 1,inplace=True)
    df_listings.drop('city', 1,inplace=True)
    df_listings.drop('state', 1,inplace=True)

    df_listings.loc[859, 'zipcode']=98122
    df_listings['zipcode'] = df_listings['zipcode'].astype("Float32").astype("Int32")
    df_listings.drop('market', 1,inplace=True)
    df_listings.drop('smart_location', 1,inplace=True)
    df_listings.drop('country_code', 1,inplace=True)
    df_listings.drop('country', 1,inplace=True)

    df_listings.loc[:, 'price'] = df_listings.loc[:, 'price'].str.strip('$')
    df_listings['price'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_listings['price'] = pd.to_numeric(df_listings['price'], 'raise', 'signed')
    df_listings['price'] = df_listings['price'].astype(dtype='Int32')

    df_listings.loc[:, 'weekly_price'] = df_listings.loc[:, 'weekly_price'].str.strip('$')
    df_listings['weekly_price'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_listings['weekly_price'] = pd.to_numeric(df_listings['weekly_price'], 'raise', 'signed')
    df_listings['weekly_price'] = df_listings['weekly_price'].astype(dtype='Int32')

    df_listings.loc[:, 'monthly_price'] = df_listings.loc[:, 'monthly_price'].str.strip('$')
    df_listings['monthly_price'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_listings['monthly_price'] = pd.to_numeric(df_listings['monthly_price'], 'raise', 'signed')
    df_listings['monthly_price'] = df_listings['monthly_price'].astype(dtype='Int32')

    df_listings.loc[:, 'security_deposit'] = df_listings.loc[:, 'security_deposit'].str.strip('$')
    df_listings['security_deposit'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_listings['security_deposit'] = pd.to_numeric(df_listings['security_deposit'], 'raise', 'signed')
    df_listings['security_deposit'] = df_listings['security_deposit'].astype(dtype='Int32')

    df_listings.loc[:, 'cleaning_fee'] = df_listings.loc[:, 'cleaning_fee'].str.strip('$')
    df_listings['cleaning_fee'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_listings['cleaning_fee'] = pd.to_numeric(df_listings['cleaning_fee'], 'raise', 'signed')
    df_listings['cleaning_fee'] = df_listings['cleaning_fee'].astype(dtype='Int32')

    df_listings['guests_included'] = pd.to_numeric(df_listings['guests_included'], 'raise', 'signed')
    df_listings['guests_included'] = df_listings['guests_included'].astype(dtype='int32')

    df_listings.loc[:, 'extra_people'] = df_listings.loc[:, 'extra_people'].str.strip('$')
    df_listings['extra_people'].replace(to_replace=r',', value="", inplace=True, regex=True)
    df_listings['extra_people'] = pd.to_numeric(df_listings['extra_people'], 'raise', 'signed')
    df_listings['extra_people'] = df_listings['extra_people'].astype(dtype='Int32')

    df_listings['minimum_nights'] = pd.to_numeric(df_listings['minimum_nights'], 'raise', 'signed')
    df_listings['minimum_nights'] = df_listings['minimum_nights'].astype(dtype='int32')

    df_listings['maximum_nights'] = pd.to_numeric(df_listings['maximum_nights'], 'raise', 'signed')
    df_listings['maximum_nights'] = df_listings['maximum_nights'].astype(dtype='int32')
    df_listings.drop('calendar_last_scraped', 1,inplace=True)

    df_listings.drop('license', 1,inplace=True)
    df_listings.drop('jurisdiction_names', 1,inplace=True)


    df_listings['security_deposit'].fillna(0, inplace=True)
    df_listings['cleaning_fee'].fillna(0, inplace=True)
    df_listings['total_price'] = df_listings['price'] + df_listings['cleaning_fee']

    df_listings["host_response_rate"] = df_listings["host_response_rate"].str.strip("%")

    df_listings["host_response_rate"].fillna(0, inplace=True)

    # df_listings["host_response_rate"] = df_listings["host_response_rate"].astype(dtype='Int32')

    # df_listings["host_response_rate"] = df_listings["host_response_rate"] / 100

    # df_listings.info()

def preprocess_reviews():
    for j in df_reviews.columns:
        df_reviews[j]=df_reviews[j].replace(to_replace={r'[^ -~]'},value='',regex=True)
    # print(df_listings["price"])
    # dic=pd.Series(df_listings["price"].values, index=df_listings["id"]).to_dict()
    # print(dic)
    df_new=df_reviews[['listing_id' , 'comments']]
    df_new=df_new.groupby('listing_id')['comments'].apply(list).reset_index(name='comments')
    df_new=df_new.rename(columns={'listing_id':'id'})
    # print(df_new)
    df_merge=pd.merge(df_listings,df_new,on='id')
    with open("data/listings_pred.csv", "w") as f:
        df_merge.to_csv(f, index=False, line_terminator='\n')

with open("data/calendar.csv") as f:
    df_calendar = pd.read_csv("data/calendar.csv")
with open("data/listings.csv") as f:
    df_listings = pd.read_csv("data/listings.csv")
with open("data/reviews.csv") as f:
    df_reviews = pd.read_csv("data/reviews.csv")

preprocess_listings()
preprocess_calendar()
preprocess_reviews()

with open("data/calendar_pred.csv", "w") as f:
    df_calendar.to_csv(f, index=False,line_terminator='\n')

# print(sys.stdout.encoding)

