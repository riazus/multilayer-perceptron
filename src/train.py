import math
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork

def sigmoid(Z):
	Z = np.clip(Z, -500, 500)
	return 1 / (1 + np.exp(-Z))


def softmax(Z):
	assert len(Z.shape) == 2
	Z_max = np.max(Z, axis=1, keepdims=1)
	e_x = np.exp(Z - Z_max)
	div = np.sum(e_x, axis=1, keepdims=1)
	return e_x / div


def sparse_categorical_cross_entropy(y_true, y_pred):
	epsilon = 1e-15
	m = y_true.shape[0]
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	return -np.sum(np.log(y_pred[np.arange(m), y_true])) / m


def he_uniform(input_size, output_size):
	limit = np.sqrt(6 / input_size)
	weight = np.random.uniform(-limit, limit, (input_size, output_size))
	return weight


def back_propagate(X, y, output, A, W):
	m = X.shape[0]
	dW, db = [], []
	dz = output.copy()
	dz[np.arange(m), y] -= 1
	dz /= m
	
	for i in reversed(range(len(W))):
		a_prev = A[i - 1] if i > 0 else X
		dW_i = np.dot(a_prev.T, dz)
		db_i = np.sum(dz, axis=0, keepdims=True)
		dW.insert(0, dW_i)
		db.insert(0, db_i)

		if i > 0:
			da = np.dot(dz, W[i].T)
			dz = da * sigmoid(A[i - 1])
	
	return dW, db


def feed_forward(X, W, b):
	A = []
	a = X

	for i in range(len(W) - 1):
		print(a)
		print("--------")
		print(W[i])
		print("--------")
		print(b[i])
		z = np.dot(a, W[i]) + b[i]
		a = sigmoid(z)
		A.append(a)

	z = np.dot(a, W[-1]) + b[-1]
	output = softmax(z)
	return output, A


def index_learning_rate(learning_rate, W, b, dW, db):
	for i in range(len(W)):
		W[i] -= learning_rate * dW[i]
		b[i] -= learning_rate * db[i]
	return W, b


def init(layers):
	weights = []
	biases = []

	for i in range(len(layers) - 1):
		input_size = layers[i]
		output_size = layers[i+1]
		biase = np.zeros((1, output_size))
		weight = he_uniform(input_size, output_size)

		weights.append(weight)
		biases.append(biase)
	
	return weights, biases


def main(layers, epochs, learning_rate, batch_size):
	weights, biases = init(layers)
	data = pd.read_csv('training-dataset.csv', header=None)

	X = data.drop(columns=0)
	y = data[0]
	X_train, X_val, y_train, y_val = train_test_split(
		X, y,
		test_size=0.2,
		stratify=y,
		random_state=42,
		shuffle=True
	)

	for epoch in range(epochs):
		for i in range(0, len(X_train), batch_size):
			batch_X = X_train[i:i+batch_size]
			batch_y = y_train[i:i+batch_size]

			output, A = feed_forward(batch_X, weights, biases)
			dWeights, dBiases = back_propagate(batch_X, batch_y, output, A, weights)
			# TODO: Update weigths and biases
			index_learning_rate(learning_rate, weights, biases, dWeights, dBiases)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		prog='Training',
		description='Train MLP on training-dataset.csv')
    
    parser.add_argument('--layer', 
                        nargs='+', type=int,
                        help='Layers and count of neurons in format: "24 24 24". By default "24 24 24"', 
                        default=[24, 24, 24],
                        required=False)
    parser.add_argument('--epochs', 
                        help='Count of the epochs. By default "80"',
                        default=80,
                        required=False)
    parser.add_argument('--learning_rate', 
                        help='Learning rate. By default 0.01',
                        default=0.01,
                        required=False)
    parser.add_argument('--batch_size', 
                        help='Feature count in batch. By default 8',
                        default=8,
                        required=False)

    args = parser.parse_args()
    main(args.layer, args.epochs, args.learning_rate, args.batch_size)