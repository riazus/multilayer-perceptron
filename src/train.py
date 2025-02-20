import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import feed_forward, sigmoid, load, get_accuracy


def plot_learning_curves(train_losses, valid_losses, 
						 train_accuracies, valid_accuracies):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    ax1.plot(train_losses, label='training Loss')
    ax1.plot(valid_losses, label='validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves - Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='train acc')
    ax2.plot(valid_accuracies, label='validation acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Learning Curves - Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def get_stopper(patience):
	counter = 0
	best_loss = None
	best_epoch = None
	improvement_threshold = 0.001
	
	def check_stopping(valid_loss, epoch):
		nonlocal counter, best_loss, best_epoch

		if best_loss is None:
			best_loss = valid_loss
			best_epoch = epoch
			return False, None
		elif valid_loss > best_loss - improvement_threshold:
			counter += 1
			if counter == patience:
				return True, best_epoch + 1
			else:
				return False, None
		else:
			best_loss = valid_loss
			best_epoch = epoch
			counter = 0
			return False, None

	return check_stopping


def get_processed_data(prefix):
	X = load(f'ressources/processed/X_{prefix}.csv')
	y = load(f'ressources/processed/y_{prefix}.csv').values.ravel()
	return y, X


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


def optimize_parameters(learning_rate, weights, biases, 
						weight_gradients, bias_gradients):
	for i in range(len(weights)):
		weights[i] -= learning_rate * weight_gradients[i]
		biases[i] -= learning_rate * bias_gradients[i]


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


def main(layers, epochs, learning_rate,
		 batch_size, early_stop_status, patience):
	train_losses = []
	valid_losses = []
	train_accuracies = []
	valid_accuracies = []
	check_early_stopping = get_stopper(patience)

	y_train, X_train = get_processed_data("train")
	y_valid, X_valid = get_processed_data("valid")

	input_layer_size = X_train.shape[1]
	output_layer_size = np.unique(y_train).shape[0]
	layer_sizes = [input_layer_size] + layers + [output_layer_size]

	weights, biases = init(layer_sizes)

	for epoch in range(epochs):
		for i in range(0, len(X_train), batch_size):
			batch_X = X_train[i:i+batch_size]
			batch_y = y_train[i:i+batch_size]

			output, activation_outputs = feed_forward(batch_X, weights, biases)
			weight_gradients, bias_gradients = back_propagate(batch_X, batch_y, 
													 output, activation_outputs, weights)

			optimize_parameters(learning_rate, weights, biases, weight_gradients, bias_gradients)

		train_output, _ = feed_forward(X_train, weights, biases)
		valid_output, _ = feed_forward(X_valid, weights, biases)

		train_loss = sparse_categorical_cross_entropy(y_train, train_output)
		valid_loss = sparse_categorical_cross_entropy(y_valid, valid_output)

		train_accuracy = get_accuracy(y_train, train_output)
		valid_accuracy = get_accuracy(y_valid, valid_output)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		train_accuracies.append(train_accuracy)
		valid_accuracies.append(valid_accuracy)

		print(f"epoch {epoch+1}/{epochs}"
                  f" - loss: {train_loss:.4f}"
                  f" - val_loss: {valid_loss:.4f}"
                  f" - acc: {train_accuracy:.4f}"
                  f" - val_acc: {valid_accuracy:.4f}")

		early_stop, best_epoch = check_early_stopping(valid_loss, epoch)
		if (early_stop == True and early_stop_status == 1):
			print(f"Early stop detected. Best epoch was: {best_epoch}.")
			break
	
	print("saving model './saved_model.npy' to disk...")
	np.save("saved_model.npy", { "weights": weights, "biases": biases })
	plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)


if __name__ == "__main__":
	def is_positive_integer(value):
		try:
			integer_value = int(value)
			if (integer_value <= 0):
				raise argparse.ArgumentTypeError(f"{value} must be a positive.")
			return integer_value
		except ValueError:
			raise argparse.ArgumentTypeError(f"{value} must be a valid integer.")
	
	def is_positive_float(value):
		try:
			float_value = float(value)
			if (float_value <= 0):
				raise argparse.ArgumentTypeError(f"{value} must be a positive.")
			return float_value
		except ValueError:
			raise argparse.ArgumentTypeError(f"{value} must be a valid float.")

	def ensure_positive_layers(value_list):
		if len(value_list) < 2:
			raise argparse.ArgumentTypeError("At least 2 layers must be specified.")
		try:
			int_list = [int(x) for x in value_list]
			if any(x <= 0 for x in int_list):
				raise argparse.ArgumentTypeError("All layer sizes must be positive integers.")
			return int_list
		except ValueError:
			raise argparse.ArgumentTypeError("All layer values must be valid integers.")

	parser = argparse.ArgumentParser(prog='MLP Training')
	parser.add_argument("-l", "--layers", 
						help='Layers and count of neurons in format: "24 24 24". By default "24 24 24"', 
						nargs='+', type=int,
						default=[24, 24, 24],
						required=False)
	parser.add_argument("-e", '--epochs', 
						help='Count of the epochs. By default 80',
						type=is_positive_integer,
						default=80,
						required=False)
	parser.add_argument("-r", '--learning_rate', 
						help='Learning rate. By default 0.0314',
						type=is_positive_float,
						default=0.0314,
						required=False)
	parser.add_argument("-s", '--batch_size', 
						help='Feature count in batch. By default 8',
						type=is_positive_integer,
						default=8,
						required=False)
	parser.add_argument('--early_stop', 
						help='Enable an early stop. By default 1',
						type=int,
						default=1,
						choices=[1, 0],
						required=False)
	parser.add_argument("-p", "--patience", 
						help='Number of epochs before early stop. By default 5',
						type=is_positive_integer,
						default=5,
						required=False)

	args = parser.parse_args()
	ensure_positive_layers(args.layers)
	main(args.layers, args.epochs, args.learning_rate, 
	  args.batch_size, args.early_stop, args.patience)