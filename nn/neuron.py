class Neuron:
    neuron_id_generator = 0

    def __init__(self, is_input, is_output):
        self.value = 0.0
        self.id = Neuron.neuron_id_generator
        Neuron.neuron_id_generator += 1
        self.is_input_neuron = is_input
        self.is_output_neuron = is_output
        self.neuron_age = 0

    def forward(self, temp_value):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError

    def fire(self):
        self.neuron_age += 1


class LinearNeuron(Neuron):
    def forward(self, temp_value):
        return temp_value

    def backward(self, output_grad):
        return 1.0
