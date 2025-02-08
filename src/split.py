import pandas as pd
import numpy as np

def minmax(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    return (X - min) / ((max - min) + 1e-15)

def main():
    val_size = 0.2
    train_size = 0.6
    df = pd.read_csv('data.csv', header=None)

    # remove first IDs column
    df = df.drop(columns=[0])
    df.columns = range(df.shape[1])

    # remove invalid rows
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)

    df[0] = df[0].map({'B': 0, 'M': 1})

    y = df.iloc[:, 0].to_frame()
    X = df.drop(columns=[0])
    X.columns = range(X.shape[1])
    X = minmax(X)

    prepared_df = pd.concat([y, X], axis=1)
    n = len(prepared_df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = prepared_df.iloc[:train_end, :]
    val = prepared_df.iloc[train_end:val_end, :]
    test = prepared_df.iloc[val_end:, :]

    train.to_csv("train-dataset.csv", index=False, header=False)
    val.to_csv("val-dataset.csv", index=False, header=False)
    test.to_csv("test-dataset.csv", index=False, header=False)

if __name__ == '__main__':
    main()