import pandas as pd
import numpy as np


def save(X, y, prefix):
    X.to_csv(f"processed/X_{prefix}.csv", index=False, header=False)
    y.to_csv(f"processed/y_{prefix}.csv", index=False, header=False)


def split_columns(df):
    y = df.iloc[:, 0].to_frame()
    X = df.drop(columns=0)
    X.columns = range(X.shape[1])
    return y, X


def z_score(X, mean, scale):
    return (X - mean) / scale


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

    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = df.iloc[:train_end, :]
    val = df.iloc[train_end:val_end, :]
    test = df.iloc[val_end:, :]

    y_train, X_train = split_columns(train)
    y_val, X_val = split_columns(val)
    y_test, X_test = split_columns(test)

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)

    # apply z_score standardization
    X_train = z_score(X_train, mean, scale)
    X_val = z_score(X_val, mean, scale)
    X_test = z_score(X_test, mean, scale)

    save(X_train, y_train, "train")
    save(X_val, y_val, "val")
    save(X_test, y_test, "test")


if __name__ == '__main__':
    main()