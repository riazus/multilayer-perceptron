from Neuron import Neuron


class NeuralNetwork:
	def __init__(self, layers):
        # Input -> Output Layer (2 neurons)
		self.input_layer = [Neuron(2), Neuron(2)]  # 2 inputs -> 2 neurons

        # Output Layer -> Hidden Layer 2 (2 neurons)
		self.hidden_layer = [Neuron(2), Neuron(2)]  # 2 inputs -> 2 neurons

        # Hidden Layer 2 -> Output Layer (1 neuron)
		self.output_neuron = Neuron(2)  # 2 inputs -> 1 neuron

	def forward(self, inputs):
        # Output Layer
		input_outputs = [neuron.forward(inputs) for neuron in self.input_layer]

        # Hidden Layer 2
		hidden_outputs2 = [neuron.forward(input_outputs) for neuron in self.hidden_layer]

        # Output Layer
		final_output = self.output_neuron.forward(hidden_outputs2)
		return final_output
