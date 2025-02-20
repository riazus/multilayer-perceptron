import numpy as np
import pandas as pd
from utils import feed_forward, load, get_accuracy, binary_cross_entropy


def main():
    saved_model = np.load("saved_model.npy", allow_pickle=True).item()
    X_test = load("ressources/processed/X_test.csv")
    y_test = load("ressources/processed/y_test.csv").values.ravel()

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

    print(f"\nSample predictions (first 15):")
    print(results.head(15))


    accuracy = get_accuracy(y_test, probabilities)
    mark = "✅" if accuracy >= 0.95 else "❌"
    print(f"Accuracy: {accuracy*100:.2f}% {mark}")

    y_pred = probabilities[:, 1]
    bce_loss = binary_cross_entropy(y_test, y_pred)
    print(f"Binary cross loss: {bce_loss}")


if __name__ == "__main__":
    main()