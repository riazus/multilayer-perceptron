import numpy as np
import pandas as pd
from utils import feed_forward, load


def get_accuracy(y_pred, y_true):
    if len(y_pred.shape) == 1:
        raise ValueError("get_accuracy(): parameters have invalid shape: (m,)")
    if y_pred.shape[1] == 1:
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true.reshape(-1, 1))
    else:
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)


def main():
    saved_model = np.load("saved_model.npy", allow_pickle=True).item()
    X_test = load("ressources/processed/X_test.csv")
    y_test = load("ressources/processed/y_test.csv").to_numpy()

    weights = saved_model['weights']
    biases = saved_model['biases']
    probabilities, _ = feed_forward(X_test, weights, biases)
    predictions = np.argmax(probabilities, axis=1).round(decimals=2)

    label_mapping = {0: 'B', 1: 'M'}
    predicted_labels = [label_mapping[pred] for pred in predictions]

    results = pd.DataFrame({
        'Prediction': predicted_labels,
        'Probability_B': np.round(probabilities[:, 0], decimals=4),
        'Probability_M': np.round(probabilities[:, 1], decimals=4)
    })

    predictions = predictions.reshape(-1, 1)
    accuracy = get_accuracy(predictions, y_test)

    print(f"\nSample predictions (first 15):")
    print(results.head(15))

    mark = "✅" if accuracy >= 0.95 else "❌"
    print(f"Accuracy: {accuracy*100:.2f}% {mark}")


if __name__ == "__main__":
    main()