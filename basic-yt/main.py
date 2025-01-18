import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("Module9Data.csv")
    # df = df.drop(df.columns[0], axis=1)
    # df.iloc[:, 0] = df.iloc[:, 0].map({'M': 0, 'B': 1})
    corr = df.corr()

    # Plot heatmap
    sb.heatmap(corr, cmap="coolwarm", annot=True)
    plt.show()

if __name__ == "__main__":
    main()
