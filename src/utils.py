import numpy as np
import pandas as pd


def softmax(Z):
	assert len(Z.shape) == 2
	Z_max = np.max(Z, axis=1, keepdims=1)
	e_x = np.exp(Z - Z_max)
	div = np.sum(e_x, axis=1, keepdims=1)
	return e_x / div


def sigmoid(Z):
	Z = np.clip(Z, -500, 500)
	return 1 / (1 + np.exp(-Z))


def feed_forward(X, weights, biases):
	activation_outputs = []
	current_activation = X

	for i in range(len(weights) - 1):
		z = np.dot(current_activation, weights[i]) + biases[i]
		current_activation = sigmoid(z)
		activation_outputs.append(current_activation)

	z = np.dot(current_activation, weights[-1]) + biases[-1]
	output = softmax(z)
	return output, activation_outputs


def load(path):
	try:
		df = pd.read_csv(path, header=None)
		return df
	except Exception:
		print(f"Error during reading csv file by the following path: {path}")
		exit(1)


def save(df, path, index=False, header=False):
	try:
		df.to_csv(path, index=index, header=header)
	except Exception:
		print(f"Error during saving csv file: {path}")
		exit(1)
	