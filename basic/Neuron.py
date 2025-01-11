import math


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


class Neuron:
	def __init__(self, num_inputs):
        # Initialize weights and bias randomly
		self.weights = [0.5] * num_inputs  # You can randomize these later
		self.bias = 0.0  # Start with bias as 0

	def forward(self, inputs):
        # Compute weighted sum (dot product + bias)
		weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        # Apply activation function
		return sigmoid(weighted_sum)