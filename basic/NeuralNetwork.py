from Neuron import Neuron


class NeuralNetwork:
	def __init__(self):
        # Input -> Hidden Layer 1 (2 neurons)
		self.hidden_layer1 = [Neuron(2), Neuron(2)]  # 2 inputs -> 2 neurons

        # Hidden Layer 1 -> Hidden Layer 2 (2 neurons)
		self.hidden_layer2 = [Neuron(2), Neuron(2)]  # 2 inputs -> 2 neurons

        # Hidden Layer 2 -> Output Layer (1 neuron)
		self.output_neuron = Neuron(2)  # 2 inputs -> 1 neuron

	def forward(self, inputs):
        # Hidden Layer 1
		hidden_outputs1 = [neuron.forward(inputs) for neuron in self.hidden_layer1]

        # Hidden Layer 2
		hidden_outputs2 = [neuron.forward(hidden_outputs1) for neuron in self.hidden_layer2]

        # Output Layer
		final_output = self.output_neuron.forward(hidden_outputs2)
		return final_output
