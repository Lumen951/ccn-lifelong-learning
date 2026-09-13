import math

import numpy as np

from networks.base import BaseLSTM

# gate block order in all weight matrices
_I, _F, _G, _O = 0, 1, 2, 3


class DenseLSTM(BaseLSTM):
    # Fully connected LSTM trained with truncated BPTT.
    # W[gate] is (hidden, input), U[gate] is (hidden, hidden), b[gate] is (hidden,).

    def __init__(self, step_size, seed, hidden_size, no_of_input_features, truncation):
        self.mt = np.random.default_rng(seed)
        self.time_step = 1
        self.step_size = step_size
        self.input_size = no_of_input_features
        self.hidden_state_size = hidden_size
        self.truncation = truncation

        self.x_queue = [np.zeros(self.input_size) for _ in range(truncation)]
        self.f_queue = [np.zeros(hidden_size) for _ in range(truncation)]
        self.g_queue = [np.zeros(hidden_size) for _ in range(truncation)]
        self.o_queue = [np.zeros(hidden_size) for _ in range(truncation)]
        self.i_queue = [np.zeros(hidden_size) for _ in range(truncation)]
        self.c_queue = [np.zeros(hidden_size) for _ in range(truncation)]
        self.h_queue = [np.zeros(hidden_size) for _ in range(truncation)]

        self.prediction_weights = np.zeros(hidden_size)
        self.prediction_weights_grad = np.zeros(hidden_size)

        bound = math.sqrt(1.0 / hidden_size)
        self.W = self.mt.uniform(-bound, bound, size=(4, hidden_size, self.input_size))
        self.U = self.mt.uniform(-bound, bound, size=(4, hidden_size, hidden_size))
        self.b = self.mt.uniform(-bound, bound, size=(4, hidden_size))
        self.W_grad = np.zeros_like(self.W)
        self.U_grad = np.zeros_like(self.U)
        self.b_grad = np.zeros_like(self.b)

    def print_features_stats(self):
        pass

    def _compute_gates(self, inputs, h_prev):
        with np.errstate(over="ignore"):
            i_temp = 1.0 / (1.0 + np.exp(-(self.W[_I] @ inputs + self.U[_I] @ h_prev + self.b[_I])))
            f_temp = 1.0 / (1.0 + np.exp(-(self.W[_F] @ inputs + self.U[_F] @ h_prev + self.b[_F])))
            g_temp = np.tanh(self.W[_G] @ inputs + self.U[_G] @ h_prev + self.b[_G])
            o_temp = 1.0 / (1.0 + np.exp(-(self.W[_O] @ inputs + self.U[_O] @ h_prev + self.b[_O])))
        return i_temp, f_temp, g_temp, o_temp

    def get_target_without_sideeffects(self, inputs):
        t_1 = (self.time_step - 1) % self.truncation
        i_temp, f_temp, g_temp, o_temp = self._compute_gates(np.asarray(inputs, dtype=float),
                                                             self.h_queue[t_1])
        c_temp = f_temp * self.c_queue[t_1] + i_temp * g_temp
        h_temp = o_temp * np.tanh(c_temp)
        return float(h_temp @ self.prediction_weights)

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        t = self.time_step % self.truncation
        t_1 = (self.time_step - 1) % self.truncation

        i_temp, f_temp, g_temp, o_temp = self._compute_gates(inputs, self.h_queue[t_1])
        self.f_queue[t] = f_temp
        self.g_queue[t] = g_temp
        self.i_queue[t] = i_temp
        self.o_queue[t] = o_temp
        self.x_queue[t] = inputs

        self.c_queue[t] = f_temp * self.c_queue[t_1] + i_temp * g_temp
        self.h_queue[t] = o_temp * np.tanh(self.c_queue[t])
        if np.isnan(self.h_queue[t]).any():
            raise FloatingPointError("NaN in hidden state")

        prediction = float(self.h_queue[t] @ self.prediction_weights)
        self.time_step += 1
        return prediction

    def get_state(self):
        t_1 = (self.time_step - 1) % self.truncation
        return list(self.h_queue[t_1]) + list(self.c_queue[t_1])

    def get_normalized_state(self):
        t = self.time_step % self.truncation
        return list(self.h_queue[t])

    def backward(self, layer):
        t = (self.time_step - 1) % self.truncation
        self.prediction_weights_grad += self.h_queue[t]

        grad_state = [self.prediction_weights.copy(), np.zeros(self.hidden_state_size)]
        for counter in range(self.truncation):
            t = (self.time_step - counter - 1) % self.truncation
            grad_state = self.backward_with_future_grad(grad_state, t)

    def backward_with_future_grad(self, grad_prev_combined, time):
        grad_f, grad_c = grad_prev_combined
        t_1 = time
        t_2 = (t_1 - 1) % self.truncation

        h_1 = self.h_queue[t_2]
        c = self.c_queue[t_1]
        c_1 = self.c_queue[t_2]
        f_t = self.f_queue[t_1]
        i_t = self.i_queue[t_1]
        g_t = self.g_queue[t_1]
        o_t = self.o_queue[t_1]
        x_t = self.x_queue[t_1]

        tanh_c = np.tanh(c)
        base = grad_f * (1 - tanh_c * tanh_c) * o_t + grad_c

        grad_i_base = base * g_t * i_t * (1 - i_t)
        grad_f_base = base * c_1 * f_t * (1 - f_t)
        grad_g_base = base * i_t * (1 - g_t * g_t)
        grad_o_base = grad_f * tanh_c * o_t * (1 - o_t)

        # outer products with x_t and h_1 accumulate into W_grad and U_grad
        for gate, grad_base in ((_I, grad_i_base), (_F, grad_f_base),
                                (_G, grad_g_base), (_O, grad_o_base)):
            self.W_grad[gate] += np.outer(grad_base, x_t)
            self.U_grad[gate] += np.outer(grad_base, h_1)
            self.b_grad[gate] += grad_base

        grad_h_new = (grad_i_base @ self.U[_I] + grad_f_base @ self.U[_F]
                      + grad_g_base @ self.U[_G] + grad_o_base @ self.U[_O])
        grad_c_new = base * f_t
        return [grad_h_new, grad_c_new]

    def decay_gradient(self, decay_rate):
        self.prediction_weights_grad *= decay_rate
        self.W_grad *= decay_rate
        self.U_grad *= decay_rate
        self.b_grad *= decay_rate

    def zero_grad(self):
        self.prediction_weights_grad[:] = 0.0
        self.W_grad[:] = 0.0
        self.U_grad[:] = 0.0
        self.b_grad[:] = 0.0

    def update_parameters(self, layer, error):
        self.prediction_weights += self.prediction_weights_grad * error * self.step_size
        self.W += self.step_size * error * self.W_grad
        self.U += self.step_size * error * self.U_grad
        self.b += self.step_size * error * self.b_grad

    def reset_state(self):
        for queue in (self.x_queue, self.f_queue, self.g_queue, self.o_queue,
                      self.i_queue, self.c_queue, self.h_queue):
            for slot in queue:
                slot[:] = 0.0


class DenseLSTMRmsProp(DenseLSTM):
    def __init__(self, step_size, seed, hidden_size, no_of_input_features,
                 truncation, beta_2, epsilon):
        super().__init__(step_size, seed, hidden_size, no_of_input_features, truncation)
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.W_grad_rmsprop = np.ones_like(self.W)
        self.U_grad_rmsprop = np.ones_like(self.U)
        self.b_grad_rmsprop = np.ones_like(self.b)
        self.prediction_weights_grad_rmsprop = np.ones_like(self.prediction_weights)

    def update_parameters(self, layer, error):
        gp = self.prediction_weights_grad * error
        self.prediction_weights_grad_rmsprop = (
            self.prediction_weights_grad_rmsprop * self.beta_2 + (1 - self.beta_2) * gp * gp)
        self.prediction_weights += gp * self.step_size / (
            np.sqrt(self.prediction_weights_grad_rmsprop) + self.epsilon)

        gw = self.W_grad * error
        self.W_grad_rmsprop = self.W_grad_rmsprop * self.beta_2 + (1 - self.beta_2) * gw * gw
        self.W += self.step_size * gw / (np.sqrt(self.W_grad_rmsprop) + self.epsilon)

        gu = self.U_grad * error
        self.U_grad_rmsprop = self.U_grad_rmsprop * self.beta_2 + (1 - self.beta_2) * gu * gu
        self.U += self.step_size * gu / (np.sqrt(self.U_grad_rmsprop) + self.epsilon)

        gb = self.b_grad * error
        self.b_grad_rmsprop = self.b_grad_rmsprop * self.beta_2 + (1 - self.beta_2) * gb * gb
        self.b += self.step_size * gb / (np.sqrt(self.b_grad_rmsprop) + self.epsilon)
