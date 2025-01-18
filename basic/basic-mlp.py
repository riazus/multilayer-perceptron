import math
from NeuralNetwork import NeuralNetwork


data = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]


def binary_cross_entropy_loss(expected, predicted):
    return -(expected * math.log(predicted) + (1 - expected) * math.log(1 - predicted))


def update_weights(weights, gradients, learning_rate):
	return [w - learning_rate * g for w, g in zip(weights, gradients)]


def sigmoid_derivative(output):
	return output * (1 - output)


def backpropagate(nn, inputs, expected):
    # Forward pass to store activations
    hidden_output1 = [neuron.forward(inputs) for neuron in nn.input_layer]
    hidden_output2 = [neuron.forward(hidden_output1) for neuron in nn.hidden_layer]
    output = nn.output_neuron.forward(hidden_output2)

    # Backward pass: Calculate deltas
    # Output layer
    output_error = output - expected
    output_delta = output_error * sigmoid_derivative(output)

    # Hidden Layer 2
    hidden2_errors = [
        output_delta * weight for weight in nn.output_neuron.weights
    ]
    hidden2_deltas = [
        error * sigmoid_derivative(hidden_output2[i])
        for i, error in enumerate(hidden2_errors)
    ]

    # Output Layer
    input_errors = [
        sum(hidden2_deltas[j] * nn.hidden_layer[j].weights[i]
            for j in range(len(nn.hidden_layer)))
        for i in range(len(nn.input_layer))
    ]
    input_deltas = [
        error * sigmoid_derivative(hidden_output1[i])
        for i, error in enumerate(input_errors)
    ]

    # Gradients for each layer
    gradients = {
        "output_neuron": (output_delta, hidden_output2),
        "hidden_layer": [(hidden2_deltas[i], hidden_output1) for i in range(len(nn.hidden_layer))],
        "input_layer": [(input_deltas[i], inputs) for i in range(len(nn.input_layer))],
    }

    return gradients



def main():
	nn = NeuralNetwork()
	learning_rate = 0.01
	num_epochs = 200000

	for epoch in range(num_epochs):
		total_loss = 0
		for row in data:
			inputs, expected = row[:2], row[2]

			# 1. Forward pass: Predict the output
			prediction = nn.forward(inputs)
			
			# 2. Compute loss: Compare prediction with expected output
			loss = binary_cross_entropy_loss(expected, prediction)
			total_loss += loss

			# 3. Backpropagation: Compute gradients
			gradients = backpropagate(nn, inputs, expected)

			# 4. Update weights using gradient descent
			for neuron, (delta, inputs) in zip(nn.input_layer, gradients["input_layer"]):
				neuron.weights = update_weights(neuron.weights, [delta * i for i in inputs], learning_rate)
				neuron.bias -= learning_rate * delta

			for neuron, (delta, inputs) in zip(nn.hidden_layer, gradients["hidden_layer"]):
				neuron.weights = update_weights(neuron.weights, [delta * i for i in inputs], learning_rate)
				neuron.bias -= learning_rate * delta

			delta, inputs = gradients["output_neuron"]
			nn.output_neuron.weights = update_weights(nn.output_neuron.weights, [delta * i for i in inputs], learning_rate)
			nn.output_neuron.bias -= learning_rate * delta

		print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")


if __name__ == "__main__":
	main()