
import csv
import sys
import pandas as pd
import numpy
import tempfile
import numpy as np
import string
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from mgwr.gwr import MGWR
import geopandas
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
from scipy.spatial import distance
from scipy import stats

def gwr_prediction(df):
    dist_lambda=lambda x: distance.euclidean(x,(-122.3380,47.6075))
    location=list(zip(df["longitude"],df["latitude"]))
    distances=list(map(dist_lambda,location))
    # distance from Seattle art museum
    df['distances']=distances
    df=df[np.abs(df['price'] - df['price'].mean()) <= (3 * df['price'].std())]

    numerical_columns=["number_of_reviews","review_scores_rating","reviews_per_month","longitude","latitude","distances"]

    X = df[numerical_columns]
    y = df['price']
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=1)
    coords_train=list(zip(X_train["longitude"],X_train["latitude"]))
    X_train=X_train.drop('longitude', 1)
    X_train=X_train.drop('latitude', 1)
    coords_test=list(zip(X_test["longitude"],X_test["latitude"]))
    X_test=X_test.drop('longitude', 1)
    X_test=X_test.drop('latitude', 1)

    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    y_train_array=y_train.values.reshape((len(y_train.values),1))

    bw=Sel_BW(coords_train,
              y_train_array,
              X_train,
              fixed=False)
    para_bw=bw.search()
    model=GWR(coords_train,
              y_train_array,
              X_train,
              para_bw,
              kernel='gaussian')
    results = model.predict(np.asarray(coords_test), X_test)
    y_test_pred = results.predy.reshape((1,len(results.predy)))[0]
    results.summary()
    # The mean squared error
    # print('Mean squared error: %.2f'
    #       % mean_squared_error(y_test, y_test_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print('Coefficient of determination: %.2f'
    #       % r2_score(y_test, y_test_pred))

    fig, ax = plt.subplots()
    n = len(y_test)

    ax.scatter(y_test.values, y_test_pred, c='tab:blue', label='tab:blue',
               alpha=0.3, edgecolors='none')
    plt.xlabel('real price')
    plt.ylabel('predicted price')
    plt.show()
    return

def lr_prediction(df):
    dist_lambda=lambda x: distance.euclidean(x,(-122.3380,47.6075))
    location=list(zip(df["longitude"],df["latitude"]))
    distances=list(map(dist_lambda,location))
    # distance from Seattle art museum
    df['distances']=distances
    df=df[np.abs(df['price'] - df['price'].mean()) <= (3 * df['price'].std())]

    numerical_columns=["host_response_rate","host_listings_count","latitude","longitude","accommodates",
                       "bathrooms","bedrooms","beds","square_feet","guests_included",
                       "extra_people","minimum_nights","maximum_nights","availability_30","availability_60",
                       "availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy",
                       "review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
                       "review_scores_value","calculated_host_listings_count","reviews_per_month","distances"]
    categorical_columns=["host_is_superhost","host_neighbourhood","host_verifications","host_has_profile_pic","host_identity_verified",
                         "neighbourhood","neighbourhood_cleansed","neighbourhood_group_cleansed","is_location_exact",
                         "property_type","room_type","calendar_updated","has_availability","requires_license","instant_bookable",
                         "cancellation_policy","require_guest_profile_picture","require_guest_phone_verification","bed_type","amenities"]

    X = df[categorical_columns + numerical_columns]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('TruncatedSVD', TruncatedSVD(n_components=30, n_iter=7, random_state=42))
    ])
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan,strategy='median')),
        ('pca',decomposition.PCA(n_components=28))
    ])
    preprocessing = ColumnTransformer(
        [
        ('cat', categorical_pipe, categorical_columns),
         ('num', numerical_pipe, numerical_columns)])


    slr = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', linear_model.LinearRegression())
    ])

    # print(X_train.isnull().values.any())

    slr.fit(X_train, y_train)

    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_test_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_test_pred))

    fig, ax = plt.subplots()
    n = len(y_test)

    ax.scatter(y_test.values, y_test_pred, c='tab:blue', label='tab:blue',
               alpha=0.3, edgecolors='none')
    plt.xlabel('real price')
    plt.ylabel('predicted price')
    plt.show()

def main():
    with open("data/trainingSet.csv") as f:
        df = pd.read_csv("data/trainingSet.csv")

    lr_prediction(df)
    gwr_prediction(df)

if __name__ == '__main__':
    main()
