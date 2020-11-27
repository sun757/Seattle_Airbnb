
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


def main():
    with open("data/listings_pred.csv") as f:
        df = pd.read_csv("data/listings_pred.csv")
    # df = df[["host_response_rate","host_is_superhost",
    #                "host_listings_count", "zipcode", "property_type", "room_type", "accommodates", "bathrooms",
    #                "bedrooms",
    #                "beds", "price", "number_of_reviews", "review_scores_rating", "cancellation_policy",
    #                "reviews_per_month"]]
    # df2=df.dropna(axis=0)
    non_num_vars = df.select_dtypes(include=['object']).columns
    df3 = df.drop(non_num_vars, axis=1)
    np.random.seed(1)

    y = df3['price']
    x = df3.drop('price', axis=1)
    imp=SimpleImputer(missing_values=np.nan,strategy='median')
    imp.fit(x)
    x=imp.transform(x)
    pca = decomposition.PCA(n_components=8)
    pca.fit(x)
    x = pca.transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    slr = linear_model.LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    print('Coefficients: \n', slr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_test_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_test_pred))

    # Plot outputs
    plt.scatter(y_test_pred, y_test)

    plt.xticks(())
    plt.yticks(())

    plt.show()
if __name__ == '__main__':
    main()
