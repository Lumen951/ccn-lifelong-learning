import math

import numpy as np

from nn.neuron import Neuron
from nn.utils import sigmoid


class LSTM(Neuron):
    # Single LSTM cell with full RTRL traces for all parameters.
    # Per-connection quantities are numpy vectors; gate order is i, f, g, o.

    def __init__(self, ui, uf, ug, uo, bi, bf, bg, bo, std_cap, decay_rate):
        super().__init__(False, False)
        self.std_cap = std_cap
        self.decay_rate = decay_rate
        self.update_statistics_flag = True
        self.epsilon = 1e-8

        self.u_i, self.u_f, self.u_g, self.u_o = ui, uf, ug, uo
        self.b_i, self.b_f, self.b_g, self.b_o = bi, bf, bg, bo

        self.c = 0.0
        self.h = 0.0
        self.old_c = 0.0
        self.old_h = 0.0
        self.i_val = 0.0
        self.f = 0.0
        self.g = 0.0
        self.o = 0.0

        # RTRL traces for recurrent weights and biases
        self.Hu_i = self.Hu_f = self.Hu_g = self.Hu_o = 0.0
        self.Cu_i = self.Cu_f = self.Cu_g = self.Cu_o = 0.0
        self.Hb_i = self.Hb_f = self.Hb_g = self.Hb_o = 0.0
        self.Cb_i = self.Cb_f = self.Cb_g = self.Cb_o = 0.0

        # gradient accumulators for recurrent weights and biases
        self.Gu_i = self.Gu_f = self.Gu_g = self.Gu_o = 0.0
        self.Gb_i = self.Gb_f = self.Gb_g = self.Gb_o = 0.0

        # squared-gradient running averages
        self.beta_2_u_i = self.beta_2_u_f = self.beta_2_u_g = self.beta_2_u_o = 1.0
        self.beta_2_b_i = self.beta_2_b_f = self.beta_2_b_g = self.beta_2_b_o = 1.0

        self.incoming_neurons = []
        self.w_i = np.zeros(0)
        self.w_f = np.zeros(0)
        self.w_g = np.zeros(0)
        self.w_o = np.zeros(0)
        self.input_means = np.zeros(0)
        self.input_std = np.zeros(0)

        self.Hw_i = np.zeros(0)
        self.Hw_f = np.zeros(0)
        self.Hw_g = np.zeros(0)
        self.Hw_o = np.zeros(0)
        self.Cw_i = np.zeros(0)
        self.Cw_f = np.zeros(0)
        self.Cw_g = np.zeros(0)
        self.Cw_o = np.zeros(0)
        self.Gw_i = np.zeros(0)
        self.Gw_f = np.zeros(0)
        self.Gw_g = np.zeros(0)
        self.Gw_o = np.zeros(0)
        self.beta_2_w_i = np.zeros(0)
        self.beta_2_w_f = np.zeros(0)
        self.beta_2_w_g = np.zeros(0)
        self.beta_2_w_o = np.zeros(0)

        self.value = 0.0

    def forward(self, temp_value):
        return temp_value

    def backward(self, output_grad):
        return 1.0

    def set_update_statistics(self, val):
        self.update_statistics_flag = val

    def fire(self):
        self.value = self.h
        self.neuron_age += 1

    def add_synapse(self, neuron, w_i, w_f, w_g, w_o):
        self.incoming_neurons.append(neuron)
        self.w_i = np.append(self.w_i, w_i)
        self.w_f = np.append(self.w_f, w_f)
        self.w_g = np.append(self.w_g, w_g)
        self.w_o = np.append(self.w_o, w_o)
        self.input_means = np.append(self.input_means, 0.0)
        self.input_std = np.append(self.input_std, 1.0)

        self.Hw_i = np.append(self.Hw_i, 0.0)
        self.Hw_f = np.append(self.Hw_f, 0.0)
        self.Hw_g = np.append(self.Hw_g, 0.0)
        self.Hw_o = np.append(self.Hw_o, 0.0)
        self.Cw_i = np.append(self.Cw_i, 0.0)
        self.Cw_f = np.append(self.Cw_f, 0.0)
        self.Cw_g = np.append(self.Cw_g, 0.0)
        self.Cw_o = np.append(self.Cw_o, 0.0)
        self.Gw_i = np.append(self.Gw_i, 0.0)
        self.Gw_f = np.append(self.Gw_f, 0.0)
        self.Gw_g = np.append(self.Gw_g, 0.0)
        self.Gw_o = np.append(self.Gw_o, 0.0)

        self.beta_2_w_i = np.append(self.beta_2_w_i, 1.0)
        self.beta_2_w_f = np.append(self.beta_2_w_f, 1.0)
        self.beta_2_w_g = np.append(self.beta_2_w_g, 1.0)
        self.beta_2_w_o = np.append(self.beta_2_w_o, 1.0)

    def get_normalized_values(self):
        values = np.array([n.value for n in self.incoming_neurons])
        return (values - self.input_means) / np.sqrt(self.input_std)

    def update_statistics(self):
        if self.update_statistics_flag:
            values = np.array([n.value for n in self.incoming_neurons])
            self.input_means = self.input_means * self.decay_rate + (1 - self.decay_rate) * values
            diff = values - self.input_means
            self.input_std = self.input_std * self.decay_rate + (1 - self.decay_rate) * diff * diff
            self.input_std = np.maximum(self.input_std, self.std_cap)

    def update_value_sync(self):
        self.old_h = self.h
        self.old_c = self.c
        x = self.get_normalized_values()

        i_val = self.b_i + self.w_i @ x + self.u_i * self.old_h
        f = self.b_f + self.w_f @ x + self.u_f * self.old_h
        g = self.b_g + self.w_g @ x + self.u_g * self.old_h
        o = self.b_o + self.w_o @ x + self.u_o * self.old_h

        self.i_val = sigmoid(i_val)
        self.f = sigmoid(f)
        self.o = sigmoid(o)
        self.g = math.tanh(g)

        self.c = self.f * self.old_c + self.i_val * self.g
        self.h = self.o * math.tanh(self.c)

    def get_hidden_state(self):
        return self.h

    def get_value_without_sideeffects(self):
        old_h = self.h
        old_c = self.c
        x = self.get_normalized_values()

        i_val = sigmoid(self.b_i + self.w_i @ x + self.u_i * old_h)
        f = sigmoid(self.b_f + self.w_f @ x + self.u_f * old_h)
        o = sigmoid(self.b_o + self.w_o @ x + self.u_o * old_h)
        g = math.tanh(self.b_g + self.w_g @ x + self.u_g * old_h)

        c = f * old_c + i_val * g
        return o * math.tanh(c)

    def compute_gradient_of_all_synapses(self):
        x = self.get_normalized_values()

        LHS_g = 1 - self.g * self.g
        LHS_f = self.f * (1 - self.f)
        LHS_i = self.i_val * (1 - self.i_val)
        LHS_o = self.o * (1 - self.o)
        tanh_of_c = math.tanh(self.c)
        dh_dc = self.o * (1 - tanh_of_c * tanh_of_c)

        # Section 1.1 -- w_i
        dg = LHS_g * (self.u_g * self.Hw_i)
        df = LHS_f * (self.u_f * self.Hw_i)
        di = LHS_i * (x + self.u_i * self.Hw_i)
        do = LHS_o * (self.u_o * self.Hw_i)
        self.Cw_i = self.f * self.Cw_i + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_i = dh_dc * self.Cw_i + tanh_of_c * do

        # Section 1.4 -- w_f
        dg = LHS_g * (self.u_g * self.Hw_f)
        df = LHS_f * (x + self.u_f * self.Hw_f)
        di = LHS_i * (self.u_i * self.Hw_f)
        do = LHS_o * (self.u_o * self.Hw_f)
        self.Cw_f = self.f * self.Cw_f + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_f = dh_dc * self.Cw_f + tanh_of_c * do

        # Section 1.5 -- w_o
        dg = LHS_g * (self.u_g * self.Hw_o)
        df = LHS_f * (self.u_f * self.Hw_o)
        di = LHS_i * (self.u_i * self.Hw_o)
        do = LHS_o * (x + self.u_o * self.Hw_o)
        self.Cw_o = self.f * self.Cw_o + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_o = dh_dc * self.Cw_o + tanh_of_c * do

        # Section 1.6 -- w_g
        dg = LHS_g * (x + self.u_g * self.Hw_g)
        df = LHS_f * (self.u_f * self.Hw_g)
        di = LHS_i * (self.u_i * self.Hw_g)
        do = LHS_o * (self.u_o * self.Hw_g)
        self.Cw_g = self.f * self.Cw_g + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_g = dh_dc * self.Cw_g + tanh_of_c * do

        # Section 1.11 -- b_f
        dg = LHS_g * (self.u_g * self.Hb_f)
        df = LHS_f * (1 + self.u_f * self.Hb_f)
        di = LHS_i * (self.u_i * self.Hb_f)
        do = LHS_o * (self.u_o * self.Hb_f)
        self.Cb_f = self.f * self.Cb_f + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_f = dh_dc * self.Cb_f + tanh_of_c * do

        # Section 1.10 -- b_g
        dg = LHS_g * (1 + self.u_g * self.Hb_g)
        df = LHS_f * (self.u_f * self.Hb_g)
        di = LHS_i * (self.u_i * self.Hb_g)
        do = LHS_o * (self.u_o * self.Hb_g)
        self.Cb_g = self.f * self.Cb_g + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_g = dh_dc * self.Cb_g + tanh_of_c * do

        # Section 1.12 -- b_o
        dg = LHS_g * (self.u_g * self.Hb_o)
        df = LHS_f * (self.u_f * self.Hb_o)
        di = LHS_i * (self.u_i * self.Hb_o)
        do = LHS_o * (1 + self.u_o * self.Hb_o)
        self.Cb_o = self.f * self.Cb_o + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_o = dh_dc * self.Cb_o + tanh_of_c * do

        # Section 1.3 -- b_i
        dg = LHS_g * (self.u_g * self.Hb_i)
        df = LHS_f * (self.u_f * self.Hb_i)
        di = LHS_i * (1 + self.u_i * self.Hb_i)
        do = LHS_o * (self.u_o * self.Hb_i)
        self.Cb_i = self.f * self.Cb_i + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_i = dh_dc * self.Cb_i + tanh_of_c * do

        # Section 1.8 -- u_f
        dg = LHS_g * (self.u_g * self.Hu_f)
        df = LHS_f * (self.old_h + self.u_f * self.Hu_f)
        di = LHS_i * (self.u_i * self.Hu_f)
        do = LHS_o * (self.u_o * self.Hu_f)
        self.Cu_f = self.f * self.Cu_f + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_f = dh_dc * self.Cu_f + tanh_of_c * do

        # Section 1.7 -- u_o
        dg = LHS_g * (self.u_g * self.Hu_o)
        df = LHS_f * (self.u_f * self.Hu_o)
        di = LHS_i * (self.u_i * self.Hu_o)
        do = LHS_o * (self.old_h + self.u_o * self.Hu_o)
        self.Cu_o = self.f * self.Cu_o + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_o = dh_dc * self.Cu_o + tanh_of_c * do

        # Section 1.9 -- u_g
        dg = LHS_g * (self.old_h + self.u_g * self.Hu_g)
        df = LHS_f * (self.u_f * self.Hu_g)
        di = LHS_i * (self.u_i * self.Hu_g)
        do = LHS_o * (self.u_o * self.Hu_g)
        self.Cu_g = self.f * self.Cu_g + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_g = dh_dc * self.Cu_g + tanh_of_c * do

        # Section 1.2 -- u_i
        dg = LHS_g * (self.u_g * self.Hu_i)
        df = LHS_f * (self.u_f * self.Hu_i)
        di = LHS_i * (self.old_h + self.u_i * self.Hu_i)
        do = LHS_o * (self.u_o * self.Hu_i)
        self.Cu_i = self.f * self.Cu_i + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_i = dh_dc * self.Cu_i + tanh_of_c * do

        self.update_statistics()

    def zero_grad(self):
        self.Gb_i = self.Gb_f = self.Gb_g = self.Gb_o = 0.0
        self.Gu_i = self.Gu_f = self.Gu_g = self.Gu_o = 0.0
        self.Gw_i[:] = 0.0
        self.Gw_f[:] = 0.0
        self.Gw_g[:] = 0.0
        self.Gw_o[:] = 0.0

    def reset_state(self):
        self.Hu_i = self.Hu_f = self.Hu_g = self.Hu_o = 0.0
        self.Cu_i = self.Cu_f = self.Cu_g = self.Cu_o = 0.0
        self.Hb_i = self.Hb_f = self.Hb_g = self.Hb_o = 0.0
        self.Cb_i = self.Cb_f = self.Cb_g = self.Cb_o = 0.0
        self.Hw_i[:] = 0.0
        self.Hw_f[:] = 0.0
        self.Hw_g[:] = 0.0
        self.Hw_o[:] = 0.0
        self.Cw_i[:] = 0.0
        self.Cw_f[:] = 0.0
        self.Cw_g[:] = 0.0
        self.Cw_o[:] = 0.0
        self.h = 0.0
        self.c = 0.0
        self.value = 0.0

    def accumulate_gradient(self, incoming_grad):
        self.Gw_f += self.Hw_f * incoming_grad
        self.Gw_i += self.Hw_i * incoming_grad
        self.Gw_o += self.Hw_o * incoming_grad
        self.Gw_g += self.Hw_g * incoming_grad
        self.Gb_f += self.Hb_f * incoming_grad
        self.Gb_i += self.Hb_i * incoming_grad
        self.Gb_o += self.Hb_o * incoming_grad
        self.Gb_g += self.Hb_g * incoming_grad
        self.Gu_f += self.Hu_f * incoming_grad
        self.Gu_i += self.Hu_i * incoming_grad
        self.Gu_o += self.Hu_o * incoming_grad
        self.Gu_g += self.Hu_g * incoming_grad

    def decay_gradient(self, decay_rate):
        self.Gw_f *= decay_rate
        self.Gw_i *= decay_rate
        self.Gw_o *= decay_rate
        self.Gw_g *= decay_rate
        self.Gb_f *= decay_rate
        self.Gb_i *= decay_rate
        self.Gb_o *= decay_rate
        self.Gb_g *= decay_rate
        self.Gu_f *= decay_rate
        self.Gu_i *= decay_rate
        self.Gu_o *= decay_rate
        self.Gu_g *= decay_rate

    def update_weights(self, step_size, error):
        b2_rate = 0.9999
        ge_f = self.Gw_f * error
        ge_i = self.Gw_i * error
        ge_o = self.Gw_o * error
        ge_g = self.Gw_g * error
        self.beta_2_w_f = b2_rate * self.beta_2_w_f + (1 - b2_rate) * ge_f * ge_f
        self.beta_2_w_i = b2_rate * self.beta_2_w_i + (1 - b2_rate) * ge_i * ge_i
        self.beta_2_w_o = b2_rate * self.beta_2_w_o + (1 - b2_rate) * ge_o * ge_o
        self.beta_2_w_g = b2_rate * self.beta_2_w_g + (1 - b2_rate) * ge_g * ge_g

        self.w_f += (step_size / (np.sqrt(self.beta_2_w_f) + self.epsilon)) * ge_f
        self.w_i += (step_size / (np.sqrt(self.beta_2_w_i) + self.epsilon)) * ge_i
        self.w_o += (step_size / (np.sqrt(self.beta_2_w_o) + self.epsilon)) * ge_o
        self.w_g += (step_size / (np.sqrt(self.beta_2_w_g) + self.epsilon)) * ge_g

        for gate in ("f", "i", "o", "g"):
            Gb = getattr(self, "Gb_" + gate) * error
            Gu = getattr(self, "Gu_" + gate) * error
            beta_b = b2_rate * getattr(self, "beta_2_b_" + gate) + (1 - b2_rate) * Gb * Gb
            beta_u = b2_rate * getattr(self, "beta_2_u_" + gate) + (1 - b2_rate) * Gu * Gu
            setattr(self, "beta_2_b_" + gate, beta_b)
            setattr(self, "beta_2_u_" + gate, beta_u)
            setattr(self, "b_" + gate,
                    getattr(self, "b_" + gate) + step_size / (math.sqrt(beta_b) + self.epsilon) * Gb)
            setattr(self, "u_" + gate,
                    getattr(self, "u_" + gate) + step_size / (math.sqrt(beta_u) + self.epsilon) * Gu)

    def update_weights_plain(self, step_size):
        self.w_f += self.Gw_f * step_size
        self.w_i += self.Gw_i * step_size
        self.w_o += self.Gw_o * step_size
        self.w_g += self.Gw_g * step_size
        self.b_f += self.Gb_f * step_size
        self.b_i += self.Gb_i * step_size
        self.b_o += self.Gb_o * step_size
        self.b_g += self.Gb_g * step_size
        self.u_f += self.Gu_f * step_size
        self.u_i += self.Gu_i * step_size
        self.u_o += self.Gu_o * step_size
        self.u_g += self.Gu_g * step_size

    def print_gradients(self):
        print("W_i gradients", self.Gw_i)
        print("W_f gradients", self.Gw_f)
        print("W_g gradients", self.Gw_g)
        print("W_o gradients", self.Gw_o)
        print("Bias", self.Gb_i, self.Gb_f, self.Gb_g, self.Gb_o)
        print("Recurrent connection", self.Gu_i, self.Gu_f, self.Gu_g, self.Gu_o)


class LSTMNormalied(LSTM):
    # LSTM cell that also normalizes its own recurrent hidden state.

    def __init__(self, ui, uf, ug, uo, bi, bf, bg, bo, std_cap, decay_rate):
        super().__init__(ui, uf, ug, uo, bi, bf, bg, bo, std_cap, decay_rate)
        self.state_mean = 0.0
        self.state_variance = 1.0

    def get_normalized_hidden_state(self, h):
        return (h - self.state_mean) / math.sqrt(self.state_variance)

    def update_statistics(self):
        super().update_statistics()
        self.state_mean = self.state_mean * self.decay_rate + (1 - self.decay_rate) * self.value
        val_minus_mean = self.value - self.state_mean
        self.state_variance = self.state_variance * self.decay_rate + (
            1 - self.decay_rate) * val_minus_mean * val_minus_mean
        if self.state_variance < self.std_cap:
            self.state_variance = self.std_cap

    def update_value_sync(self):
        self.old_h = self.h
        self.old_c = self.c
        x = self.get_normalized_values()
        normalized_old_h = self.get_normalized_hidden_state(self.old_h)

        i_val = self.b_i + self.w_i @ x + self.u_i * normalized_old_h
        f = self.b_f + self.w_f @ x + self.u_f * normalized_old_h
        g = self.b_g + self.w_g @ x + self.u_g * normalized_old_h
        o = self.b_o + self.w_o @ x + self.u_o * normalized_old_h

        self.i_val = sigmoid(i_val)
        self.f = sigmoid(f)
        self.o = sigmoid(o)
        self.g = math.tanh(g)

        self.c = self.f * self.old_c + self.i_val * self.g
        self.h = self.o * math.tanh(self.c)

    def get_value_without_sideeffects(self):
        old_h = self.get_normalized_hidden_state(self.h)
        old_c = self.c
        x = self.get_normalized_values()

        i_val = sigmoid(self.b_i + self.w_i @ x + self.u_i * old_h)
        f = sigmoid(self.b_f + self.w_f @ x + self.u_f * old_h)
        o = sigmoid(self.b_o + self.w_o @ x + self.u_o * old_h)
        g = math.tanh(self.b_g + self.w_g @ x + self.u_g * old_h)

        c = f * old_c + i_val * g
        return o * math.tanh(c)

    def compute_gradient_of_all_synapses(self):
        x = self.get_normalized_values()
        inv_std = 1.0 / math.sqrt(self.state_variance)

        LHS_g = 1 - self.g * self.g
        LHS_f = self.f * (1 - self.f)
        LHS_i = self.i_val * (1 - self.i_val)
        LHS_o = self.o * (1 - self.o)
        tanh_of_c = math.tanh(self.c)
        dh_dc = self.o * (1 - tanh_of_c * tanh_of_c)

        # Section 1.1 -- w_i
        dg = LHS_g * (self.u_g * (self.Hw_i * inv_std))
        df = LHS_f * (self.u_f * (self.Hw_i * inv_std))
        di = LHS_i * (x + self.u_i * (self.Hw_i * inv_std))
        do = LHS_o * (self.u_o * (self.Hw_i * inv_std))
        self.Cw_i = self.f * self.Cw_i + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_i = dh_dc * self.Cw_i + tanh_of_c * do

        # Section 1.4 -- w_f
        dg = LHS_g * (self.u_g * (self.Hw_f * inv_std))
        df = LHS_f * (x + self.u_f * (self.Hw_f * inv_std))
        di = LHS_i * (self.u_i * (self.Hw_f * inv_std))
        do = LHS_o * (self.u_o * (self.Hw_f * inv_std))
        self.Cw_f = self.f * self.Cw_f + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_f = dh_dc * self.Cw_f + tanh_of_c * do

        # Section 1.5 -- w_o
        dg = LHS_g * (self.u_g * (self.Hw_o * inv_std))
        df = LHS_f * (self.u_f * (self.Hw_o * inv_std))
        di = LHS_i * (self.u_i * (self.Hw_o * inv_std))
        do = LHS_o * (x + self.u_o * (self.Hw_o * inv_std))
        self.Cw_o = self.f * self.Cw_o + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_o = dh_dc * self.Cw_o + tanh_of_c * do

        # Section 1.6 -- w_g
        dg = LHS_g * (x + self.u_g * (self.Hw_g * inv_std))
        df = LHS_f * (self.u_f * (self.Hw_g * inv_std))
        di = LHS_i * (self.u_i * (self.Hw_g * inv_std))
        do = LHS_o * (self.u_o * (self.Hw_g * inv_std))
        self.Cw_g = self.f * self.Cw_g + self.old_c * df + self.i_val * dg + self.g * di
        self.Hw_g = dh_dc * self.Cw_g + tanh_of_c * do

        # Section 1.11 -- b_f
        dg = LHS_g * (self.u_g * (self.Hb_f * inv_std))
        df = LHS_f * (1 + self.u_f * (self.Hb_f * inv_std))
        di = LHS_i * (self.u_i * (self.Hb_f * inv_std))
        do = LHS_o * (self.u_o * (self.Hb_f * inv_std))
        self.Cb_f = self.f * self.Cb_f + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_f = dh_dc * self.Cb_f + tanh_of_c * do

        # Section 1.10 -- b_g
        dg = LHS_g * (1 + self.u_g * (self.Hb_g * inv_std))
        df = LHS_f * (self.u_f * (self.Hb_g * inv_std))
        di = LHS_i * (self.u_i * (self.Hb_g * inv_std))
        do = LHS_o * (self.u_o * (self.Hb_g * inv_std))
        self.Cb_g = self.f * self.Cb_g + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_g = dh_dc * self.Cb_g + tanh_of_c * do

        # Section 1.12 -- b_o
        dg = LHS_g * (self.u_g * (self.Hb_o * inv_std))
        df = LHS_f * (self.u_f * (self.Hb_o * inv_std))
        di = LHS_i * (self.u_i * (self.Hb_o * inv_std))
        do = LHS_o * (1 + self.u_o * (self.Hb_o * inv_std))
        self.Cb_o = self.f * self.Cb_o + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_o = dh_dc * self.Cb_o + tanh_of_c * do

        # Section 1.3 -- b_i
        dg = LHS_g * (self.u_g * (self.Hb_i * inv_std))
        df = LHS_f * (self.u_f * (self.Hb_i * inv_std))
        di = LHS_i * (1 + self.u_i * (self.Hb_i * inv_std))
        do = LHS_o * (self.u_o * (self.Hb_i * inv_std))
        self.Cb_i = self.f * self.Cb_i + self.old_c * df + self.i_val * dg + self.g * di
        self.Hb_i = dh_dc * self.Cb_i + tanh_of_c * do

        normalized_old_h = self.get_normalized_hidden_state(self.old_h)

        # Section 1.8 -- u_f
        dg = LHS_g * (self.u_g * (self.Hu_f * inv_std))
        df = LHS_f * (normalized_old_h + self.u_f * (self.Hu_f * inv_std))
        di = LHS_i * (self.u_i * (self.Hu_f * inv_std))
        do = LHS_o * (self.u_o * (self.Hu_f * inv_std))
        self.Cu_f = self.f * self.Cu_f + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_f = dh_dc * self.Cu_f + tanh_of_c * do

        # Section 1.7 -- u_o
        dg = LHS_g * (self.u_g * (self.Hu_o * inv_std))
        df = LHS_f * (self.u_f * (self.Hu_o * inv_std))
        di = LHS_i * (self.u_i * (self.Hu_o * inv_std))
        do = LHS_o * (normalized_old_h + self.u_o * (self.Hu_o * inv_std))
        self.Cu_o = self.f * self.Cu_o + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_o = dh_dc * self.Cu_o + tanh_of_c * do

        # Section 1.9 -- u_g
        dg = LHS_g * (normalized_old_h + self.u_g * (self.Hu_g * inv_std))
        df = LHS_f * (self.u_f * (self.Hu_g * inv_std))
        di = LHS_i * (self.u_i * (self.Hu_g * inv_std))
        do = LHS_o * (self.u_o * (self.Hu_g * inv_std))
        self.Cu_g = self.f * self.Cu_g + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_g = dh_dc * self.Cu_g + tanh_of_c * do

        # Section 1.2 -- u_i
        dg = LHS_g * (self.u_g * (self.Hu_i * inv_std))
        df = LHS_f * (self.u_f * (self.Hu_i * inv_std))
        di = LHS_i * (normalized_old_h + self.u_i * (self.Hu_i * inv_std))
        do = LHS_o * (self.u_o * (self.Hu_i * inv_std))
        self.Cu_i = self.f * self.Cu_i + self.old_c * df + self.i_val * dg + self.g * di
        self.Hu_i = dh_dc * self.Cu_i + tanh_of_c * do

        self.update_statistics()
