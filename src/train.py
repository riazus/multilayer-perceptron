import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    ax1.plot(train_losses, label='training Loss')
    ax1.plot(val_losses, label='validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves - Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='train acc')
    ax2.plot(val_accuracies, label='validation acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Learning Curves - Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def get_processed_data(prefix):
	X = pd.read_csv(f'ressources/processed/X_{prefix}.csv', header=None)
	y = pd.read_csv(f'ressources/processed/y_{prefix}.csv', header=None).values.ravel()
	return y, X


def sigmoid(Z):
	Z = np.clip(Z, -500, 500)
	return 1 / (1 + np.exp(-Z))


def get_accuracy(y_pred, y_true):
    if len(y_pred.shape) == 1:
        raise ValueError("get_accuracy(): parameters have invalid shape: (m,)")
    if y_pred.shape[1] == 1:
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true.reshape(-1, 1))
    else:
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)


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


def back_propagate(X, y, output, activation_outputs, weights):
	m = X.shape[0]
	weight_gradients = []
	bias_gradients = []
	grad_output = output.copy()
	grad_output[np.arange(m), y] -= 1
	grad_output /= m
	
	for i in reversed(range(len(weights))):
		previous_activation = activation_outputs[i - 1] if i > 0 else X
		weight_gradient = np.dot(previous_activation.T, grad_output)
		bias_gradient = np.sum(grad_output, axis=0, keepdims=True)
		weight_gradients.insert(0, weight_gradient)
		bias_gradients.insert(0, bias_gradient)

		if i > 0:
			grad_activation = np.dot(grad_output, weights[i].T)
			sigmoid_activation = sigmoid(activation_outputs[i - 1])
			grad_output = grad_activation * sigmoid_activation * (1 - sigmoid_activation)
	
	return weight_gradients, bias_gradients


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


def index_learning_rate(learning_rate, weights, biases, 
						weight_gradients, bias_gradients):
	for i in range(len(weights)):
		weights[i] -= learning_rate * weight_gradients[i]
		biases[i] -= learning_rate * bias_gradients[i]
	return weights, biases


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
	train_losses = []
	val_losses = []
	train_accuracies = []
	val_accuracies = []

	y_train, X_train = get_processed_data("train")
	y_val, X_val = get_processed_data("val")

	input_layer_size = X_train.shape[1]
	output_layer_size = np.unique(y_train).shape[0]
	layer_sizes = [input_layer_size] + layers + [output_layer_size]

	weights, biases = init(layer_sizes)

	for epoch in range(epochs):
		for i in range(0, len(X_train), batch_size):
			batch_X = X_train[i:i+batch_size]
			batch_y = y_train[i:i+batch_size]

			output, A = feed_forward(batch_X, weights, biases)
			weight_gradients, bias_gradients = back_propagate(batch_X, batch_y, output, A, weights)
			weights, biases = index_learning_rate(learning_rate, 
													weights, biases, 
													weight_gradients, bias_gradients)
		
		train_output, _ = feed_forward(X_train, weights, biases)
		val_output, _ = feed_forward(X_val, weights, biases)

		train_loss = sparse_categorical_cross_entropy(y_train, train_output)
		val_loss = sparse_categorical_cross_entropy(y_val, val_output)

		train_accuracy = get_accuracy(train_output, y_train)
		val_accuracy = get_accuracy(val_output, y_val)

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		train_accuracies.append(train_accuracy)
		val_accuracies.append(val_accuracy)

		print(f"epoch {epoch+1}/{epochs}"
                  f" - loss: {train_loss:.4f}"
                  f" - val_loss: {val_loss:.4f}"
                  f" - acc: {train_accuracy:.4f}"
                  f" - val_acc: {val_accuracy:.4f}")
	
	plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
	return weights, biases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Training')
    parser.add_argument('--layers', 
                        help='Layers and count of neurons in format: "24 24 24". By default "24 24 24"', 
                        nargs='+', type=int,
                        default=[24, 24, 24],
                        required=False)
    parser.add_argument('--epochs', 
                        help='Count of the epochs. By default "80"',
						type=int,
                        default=80,
                        required=False)
    parser.add_argument('--learning_rate', 
                        help='Learning rate. By default 0.0314',
                        type=float,
						default=0.0314,
                        required=False)
    parser.add_argument('--batch_size', 
                        help='Feature count in batch. By default 8',
                        type=int,
						default=8,
                        required=False)

    args = parser.parse_args()
    main(args.layers, args.epochs, args.learning_rate, args.batch_size)