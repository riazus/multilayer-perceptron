import pandas as pd
import numpy as np


def save(X, y, prefix):
    path_X = f"ressources/processed/X_{prefix}.csv"
    path_y = f"ressources/processed/y_{prefix}.csv"
    X.to_csv(path_X, index=False, header=False)
    y.to_csv(path_y, index=False, header=False)


def split_columns(df):
    y = df.iloc[:, 0].to_frame()
    X = df.drop(columns=0)
    X.columns = range(X.shape[1])
    return y, X


def z_score(X, mean, scale):
    return (X - mean) / scale


def main():
    valid_size = 0.2
    train_size = 0.6
    df = pd.read_csv('ressources/raw/data.csv', header=None)

    # remove first IDs column
    df = df.drop(columns=[0])
    df.columns = range(df.shape[1])

    # remove invalid rows
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)

    df[0] = df[0].map({'B': 0, 'M': 1})

    n = len(df)
    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))

    train = df.iloc[:train_end, :]
    valid = df.iloc[train_end:valid_end, :]
    test = df.iloc[valid_end:, :]

    y_train, X_train = split_columns(train)
    y_valid, X_valid = split_columns(valid)
    y_test, X_test = split_columns(test)

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)

    # apply z_score standardization
    X_train = z_score(X_train, mean, scale)
    X_valid = z_score(X_valid, mean, scale)
    X_test = z_score(X_test, mean, scale)

    save(X_train, y_train, "train")
    save(X_valid, y_valid, "valid")
    save(X_test, y_test, "test")


if __name__ == '__main__':
    main()