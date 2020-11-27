import sys
import pandas as pd
import numpy as np

def main():
    with open("data/listings_pred.csv") as f:
        df = pd.read_csv("data/listings_pred.csv")

    test_set = df.sample(random_state=47, frac=0.2)
    training_set = df.loc[~df.index.isin(test_set.index)]

    with open("data/trainingSet.csv", "w") as f:
        training_set.to_csv(f, index=False, line_terminator='\n')

    with open("data/testSet.csv", "w") as f:
        test_set.to_csv(f, index=False, line_terminator='\n')

if __name__ == '__main__':
    main()
