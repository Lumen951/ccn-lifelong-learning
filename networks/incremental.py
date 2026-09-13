import math

import numpy as np

from nn.neuron import LinearNeuron
from nn.lstm import LSTM, LSTMNormalied
from networks.base import BaseLSTM


class IncrementalNetworks(BaseLSTM):
    # Columnar/constructive network: a set of LSTM cells organized in layers,
    # read out by a normalized linear predictor. Gradient traces accumulate
    # online (RTRL-style) and are decayed by lambda*gamma each step.

    def __init__(self, step_size, seed, no_of_input_features, total_targets,
                 total_recurrent_features, layer_size, std_cap, decay_rate):
        self.decay_rate = decay_rate
        self.layer_size = int(layer_size)
        self.step_size = step_size
        self.std_cap = std_cap
        self.mt = np.random.default_rng(seed)
        second_mt = np.random.default_rng(seed)

        self.input_neurons = [LinearNeuron(True, False) for _ in range(no_of_input_features)]
        self.LSTM_neurons = []
        self.indexes_lstm_cells = []
        self._build_cells(total_recurrent_features, no_of_input_features,
                          weight_range=0.01, index_draws=50000,
                          restrict_to_lower_layers=True, cell_type=LSTM,
                          cell_decay=decay_rate, second_mt=second_mt,
                          verbose=True)

        self.predictions = 0.0
        self.bias = 0.0
        self.bias_gradients = 0.0
        self.bias_beta_2 = 1.0
        n = len(self.LSTM_neurons)
        self.prediction_weights = np.zeros(n)
        self.prediction_weights_gradient = np.zeros(n)
        self.avg_feature_value = np.zeros(n)
        self.feature_mean = np.zeros(n)
        self.feature_std = np.ones(n)
        self.prediction_weights_beta_2 = np.ones(n)

    def _build_cells(self, total_recurrent_features, no_of_input_features,
                     weight_range, index_draws, restrict_to_lower_layers,
                     cell_type, cell_decay, second_mt, verbose=False):
        for i in range(total_recurrent_features):
            cell = cell_type(
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.mt.uniform(-weight_range, weight_range),
                self.std_cap, cell_decay)
            self.indexes_lstm_cells.append(i)
            self.LSTM_neurons.append(cell)

        total = no_of_input_features + total_recurrent_features
        for counter in range(total_recurrent_features):
            layer_no = counter // self.layer_size
            map_index = np.zeros(total, dtype=bool)
            total_features = 0
            for index in second_mt.integers(0, total, size=index_draws):
                if not map_index[index]:
                    map_index[index] = True
                    if index < no_of_input_features:
                        total_features += 1
                        neuron_ref = self.input_neurons[index]
                        self.LSTM_neurons[counter].add_synapse(
                            neuron_ref,
                            self.mt.uniform(-weight_range, weight_range),
                            self.mt.uniform(-weight_range, weight_range),
                            self.mt.uniform(-weight_range, weight_range),
                            self.mt.uniform(-weight_range, weight_range))
                    else:
                        lstm_index = index - no_of_input_features
                        new_layer_no = lstm_index // self.layer_size
                        if not restrict_to_lower_layers or new_layer_no < layer_no:
                            total_features += 1
                            neuron_ref = self.LSTM_neurons[lstm_index]
                            self.LSTM_neurons[counter].add_synapse(
                                neuron_ref,
                                self.mt.uniform(-weight_range, weight_range),
                                self.mt.uniform(-weight_range, weight_range),
                                self.mt.uniform(-weight_range, weight_range),
                                self.mt.uniform(-weight_range, weight_range))
            if verbose:
                print("Total features =", total_features)

    def print_features_stats(self):
        for counter, cell in enumerate(self.LSTM_neurons):
            print("Counter =", counter)
            print("Feature mean =", self.feature_mean[counter])
            print("Feature std =", self.feature_std[counter])
            print("feature value =", cell.value)

    def forward(self, inputs):
        for i, v in enumerate(inputs):
            self.input_neurons[i].value = float(v)

        for cell in self.LSTM_neurons:
            cell.update_value_sync()

        for counter, cell in enumerate(self.LSTM_neurons):
            cell.fire()
            self.avg_feature_value[counter] = (
                self.avg_feature_value[counter] * self.decay_rate
                + (1 - self.decay_rate) * (cell.value - self.feature_mean[counter]))

        for counter, cell in enumerate(self.LSTM_neurons):
            self.feature_mean[counter] = (
                self.feature_mean[counter] * self.decay_rate
                + (1 - self.decay_rate) * cell.value)
            temp = self.feature_mean[counter] - cell.value
            self.feature_std[counter] = (
                self.feature_std[counter] * self.decay_rate
                + (1 - self.decay_rate) * temp * temp)
            if self.feature_std[counter] < self.std_cap:
                self.feature_std[counter] = self.std_cap

        normalized = (np.array([c.value for c in self.LSTM_neurons]) - self.feature_mean) / np.sqrt(
            self.feature_std)
        self.predictions = float(self.prediction_weights @ normalized) + self.bias
        return self.predictions

    def get_target_without_sideeffects(self, inputs):
        backup = [self.input_neurons[i].value for i in range(len(inputs))]
        for i, v in enumerate(inputs):
            self.input_neurons[i].value = float(v)

        hidden_state = np.array([cell.get_value_without_sideeffects()
                                 for cell in self.LSTM_neurons])
        temp_prediction = float(
            self.prediction_weights @ ((hidden_state - self.feature_mean) / np.sqrt(self.feature_std)))
        temp_prediction += self.bias

        for i in range(len(inputs)):
            self.input_neurons[i].value = backup[i]
        return temp_prediction

    def reset_state(self):
        for cell in self.LSTM_neurons:
            cell.reset_state()

    def zero_grad(self):
        for cell in self.LSTM_neurons:
            cell.zero_grad()
        self.prediction_weights_gradient[:] = 0.0
        self.bias_gradients = 0.0

    def decay_gradient(self, decay_rate):
        for cell in self.LSTM_neurons:
            cell.decay_gradient(decay_rate)
        self.prediction_weights_gradient *= decay_rate
        self.bias_gradients *= decay_rate

    def get_prediction_weights(self):
        return list(self.prediction_weights)

    def get_prediction_gradients(self):
        return list(self.prediction_weights_gradient)

    def get_state(self):
        return [cell.value for cell in self.LSTM_neurons]

    def get_normalized_state(self):
        values = np.array([c.value for c in self.LSTM_neurons])
        return list((values - self.feature_mean) / np.sqrt(self.feature_std))

    def backward(self, layer):
        lo = layer * self.layer_size
        hi = (layer + 1) * self.layer_size
        for index in range(lo, min(hi, len(self.LSTM_neurons))):
            cell = self.LSTM_neurons[index]
            cell.compute_gradient_of_all_synapses()
            cell.accumulate_gradient(
                float(self.prediction_weights[index] / math.sqrt(self.feature_std[index])))

        values = np.array([c.value for c in self.LSTM_neurons])
        self.prediction_weights_gradient += (values - self.feature_mean) / np.sqrt(self.feature_std)
        self.bias_gradients += 1.0

    def update_parameters(self, layer, error):
        b2_decay = 0.9999
        epsilon = 1e-8
        lo = layer * self.layer_size
        hi = (layer + 1) * self.layer_size
        for index, cell in enumerate(self.LSTM_neurons):
            if lo <= index < hi:
                cell.update_weights(self.step_size, error)
            elif index < lo:
                cell.set_update_statistics(False)

        # normalize the step-size by the number of active readout features
        temp_step_size = self.step_size / hi
        active = min(hi, len(self.LSTM_neurons))
        g = self.prediction_weights_gradient[:active] * error
        self.prediction_weights_beta_2[:active] = (
            self.prediction_weights_beta_2[:active] * b2_decay + (1 - b2_decay) * g * g)
        self.prediction_weights[:active] += g * (
            temp_step_size / (np.sqrt(self.prediction_weights_beta_2[:active]) + epsilon))

        gb = error * self.bias_gradients
        self.bias_beta_2 = self.bias_beta_2 * b2_decay + (1 - b2_decay) * gb * gb
        self.bias += (temp_step_size / (math.sqrt(self.bias_beta_2) + epsilon)) * self.bias_gradients * error

    def update_parameters_no_freeze(self, error):
        for cell in self.LSTM_neurons:
            cell.update_weights(self.step_size, error)
        self.prediction_weights += self.prediction_weights_gradient * error * self.step_size
        self.bias += error * self.step_size * self.bias_gradients

    def read_output_values(self):
        return self.predictions


class Snap1(IncrementalNetworks):
    def __init__(self, step_size, seed, no_of_input_features, total_targets,
                 total_recurrent_features, layer_size, std_cap):
        self.decay_rate = 0.99999
        self.layer_size = int(layer_size)
        self.step_size = step_size
        self.std_cap = std_cap
        self.mt = np.random.default_rng(seed)
        second_mt = np.random.default_rng(seed)

        self.input_neurons = [LinearNeuron(True, False) for _ in range(no_of_input_features)]
        self.LSTM_neurons = []
        self.indexes_lstm_cells = []
        self._build_cells(total_recurrent_features, no_of_input_features,
                          weight_range=0.1, index_draws=10000,
                          restrict_to_lower_layers=False, cell_type=LSTM,
                          cell_decay=0.99999, second_mt=second_mt)

        self.predictions = 0.0
        self.bias = 0.0
        self.bias_gradients = 0.0
        self.bias_beta_2 = 1.0
        n = len(self.LSTM_neurons)
        self.prediction_weights = np.zeros(n)
        self.prediction_weights_gradient = np.zeros(n)
        self.avg_feature_value = np.zeros(n)
        self.feature_mean = np.zeros(n)
        self.feature_std = np.ones(n)
        self.prediction_weights_beta_2 = np.ones(n)

    def backward(self, layer):
        for index, cell in enumerate(self.LSTM_neurons):
            cell.compute_gradient_of_all_synapses()
            cell.accumulate_gradient(
                float(self.prediction_weights[index] / math.sqrt(self.feature_std[index])))

        values = np.array([c.value for c in self.LSTM_neurons])
        self.prediction_weights_gradient += (values - self.feature_mean) / np.sqrt(self.feature_std)
        self.bias_gradients += 1.0

    def update_parameters(self, layer, error):
        for cell in self.LSTM_neurons:
            cell.update_weights(self.step_size, error)

        total_features_for_prediction = (layer + 1) * self.layer_size
        self.prediction_weights += (
            self.prediction_weights_gradient * error
            * (self.step_size / total_features_for_prediction))
        self.bias += error * self.step_size * self.bias_gradients


class NormalizedIncrementalNetworks(IncrementalNetworks):
    def __init__(self, step_size, seed, no_of_input_features, total_targets,
                 total_recurrent_features, layer_size, std_cap, decay_rate):
        self.decay_rate = decay_rate
        self.layer_size = int(layer_size)
        self.step_size = step_size
        self.std_cap = std_cap
        self.mt = np.random.default_rng(seed)
        second_mt = np.random.default_rng(seed)

        self.input_neurons = [LinearNeuron(True, False) for _ in range(no_of_input_features)]
        self.LSTM_neurons = []
        self.indexes_lstm_cells = []
        self._build_cells(total_recurrent_features, no_of_input_features,
                          weight_range=0.1, index_draws=10000,
                          restrict_to_lower_layers=True, cell_type=LSTMNormalied,
                          cell_decay=decay_rate, second_mt=second_mt)

        self.predictions = 0.0
        self.bias = 0.0
        self.bias_gradients = 0.0
        self.bias_beta_2 = 1.0
        n = len(self.LSTM_neurons)
        self.prediction_weights = np.zeros(n)
        self.prediction_weights_gradient = np.zeros(n)
        self.avg_feature_value = np.zeros(n)
        self.feature_mean = np.zeros(n)
        self.feature_std = np.ones(n)
        self.prediction_weights_beta_2 = np.ones(n)
