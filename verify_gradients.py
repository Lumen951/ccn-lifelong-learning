"""Finite-difference verification of the ported gradient computations.

Checks that the RTRL traces in nn/lstm.py produce gradients matching
central finite differences of the replayed hidden state, and that the
truncated-BPTT gradients in networks/bptt.py match finite differences of
the replayed windowed prediction sum.
"""
import copy

import numpy as np

from nn.neuron import LinearNeuron
from nn.lstm import LSTM
from networks.bptt import DenseLSTM


def central_diff(f, ref, eps=1e-5):
    orig = ref.get()
    ref.set(orig + eps)
    hi = f.value()
    ref.set(orig - eps)
    lo = f.value()
    ref.set(orig)
    return (hi - lo) / (2 * eps)


class Ref:
    # mutable handle to a scalar parameter (numpy element or object attr)

    def __init__(self, getter, setter):
        self._g = getter
        self._s = setter

    def get(self):
        return self._g()

    def set(self, v):
        self._s(v)


class CellProbe:
    # Wraps a single LSTM cell driven by fixed inputs; replays T steps from a
    # reset state so hidden state (and therefore the RTRL traces) reproduce.

    def __init__(self, seed=0, n_inputs=3, T=6):
        rng = np.random.default_rng(seed)
        self.T = T
        self.inputs = rng.integers(0, 2, size=(T, n_inputs)).astype(float)
        self.input_neurons = [LinearNeuron(True, False) for _ in range(n_inputs)]

        def r():
            return float(rng.uniform(-0.01, 0.01))

        self.cell = LSTM(r(), r(), r(), r(), r(), r(), r(), r(), std_cap=0.1, decay_rate=0.999)
        for n in self.input_neurons:
            self.cell.add_synapse(n, r(), r(), r(), r())
        # snapshot input normalization stats so every replay starts identical
        self._init_means = self.cell.input_means.copy()
        self._init_std = self.cell.input_std.copy()
        self.names = []
        for k in range(n_inputs):
            for gate in ("i", "f", "g", "o"):
                self.names.append(("w_" + gate, k))
        for p in ("u_i", "u_f", "u_g", "u_o", "b_i", "b_f", "b_g", "b_o"):
            self.names.append((p, None))

    def _replay(self):
        cell = self.cell
        cell.input_means = self._init_means.copy()
        cell.input_std = self._init_std.copy()
        cell.reset_state()
        for t in range(self.T):
            for n, v in zip(self.input_neurons, self.inputs[t]):
                n.value = v
            cell.update_value_sync()
            cell.fire()
            cell.compute_gradient_of_all_synapses()
        return cell.h

    def value(self):
        return self._replay()

    @property
    def params(self):
        c = self.cell
        out = {}
        for k in range(len(self.input_neurons)):
            for gate in ("i", "f", "g", "o"):
                arr = getattr(c, "w_" + gate)
                out[("w_" + gate, k)] = Ref(lambda a=arr, i=k: a[i],
                                            lambda v, a=arr, i=k: a.__setitem__(i, v))
        for p in ("u_i", "u_f", "u_g", "u_o", "b_i", "b_f", "b_g", "b_o"):
            out[(p, None)] = Ref(lambda n=p: getattr(c, n),
                                 lambda v, n=p: setattr(c, n, v))
        return out

    def analytic_grad(self):
        # replay, then read d(h_T)/d(theta) off the traces after a single
        # accumulate with unit incoming gradient
        self._replay()
        cell = self.cell
        cell.accumulate_gradient(1.0)
        out = {}
        for k in range(len(self.input_neurons)):
            for gate in ("i", "f", "g", "o"):
                out[("w_" + gate, k)] = getattr(cell, "Gw_" + gate)[k]
        for p in ("u_i", "u_f", "u_g", "u_o", "b_i", "b_f", "b_g", "b_o"):
            out[(p, None)] = getattr(cell, "G" + p)
        return out


def check_lstm():
    probe = CellProbe()
    analytic = probe.analytic_grad()
    worst = 0.0
    for name in probe.names:
        fd = central_diff(probe, probe.params[name])
        an = analytic[name]
        scale = max(1.0, abs(fd), abs(an))
        rel = abs(fd - an) / scale
        worst = max(worst, rel)
        assert rel < 1e-3, f"LSTM grad mismatch at {name}: fd={fd} analytic={an}"
    print(f"LSTM RTRL gradient check: PASS (worst rel err {worst:.2e}, "
          f"{len(probe.names)} params)")


class BPTTProbe:
    def __init__(self, seed=0, input_size=3, hidden=4, trunc=5, T=7):
        self.net = DenseLSTM(step_size=0.01, seed=seed, hidden_size=hidden,
                             no_of_input_features=input_size, truncation=trunc)
        rng = np.random.default_rng(seed + 1)
        self.inputs = rng.integers(0, 2, size=(T, input_size)).astype(float)
        self.T = T

    def _fresh(self):
        net = self.net
        net.reset_state()
        net.time_step = 1

    def run_backward(self):
        net = self.net
        net.zero_grad()
        self._fresh()
        for t in range(self.T):
            net.forward(self.inputs[t])
            net.backward(0)

    def total_pred_sum(self):
        net = self.net
        self._fresh()
        return float(sum(net.forward(self.inputs[t]) for t in range(self.T)))

    @property
    def params(self):
        net = self.net
        return {"W": net.W, "U": net.U, "b": net.b, "pred": net.prediction_weights}


def check_bptt():
    probe = BPTTProbe()
    probe.run_backward()
    net = probe.net
    targets = [("W", net.W_grad), ("U", net.U_grad), ("b", net.b_grad),
               ("pred", net.prediction_weights_grad)]
    worst = 0.0
    checked = 0
    rng = np.random.default_rng(7)
    for name, grad_arr in targets:
        flat_grad = np.asarray(grad_arr).ravel()
        idxs = rng.choice(len(flat_grad), size=min(20, len(flat_grad)), replace=False)
        for idx in idxs:
            orig = probe.params[name].ravel()[idx]

            def val():
                return probe.total_pred_sum()

            eps = 1e-4
            probe.params[name].ravel()[idx] = orig + eps
            hi = val()
            probe.params[name].ravel()[idx] = orig - eps
            lo = val()
            probe.params[name].ravel()[idx] = orig
            fd = (hi - lo) / (2 * eps)
            an = flat_grad[idx]
            scale = max(1.0, abs(fd), abs(an))
            rel = abs(fd - an) / scale
            worst = max(worst, rel)
            checked += 1
            assert rel < 1e-3, f"BPTT grad mismatch {name}[{idx}]: fd={fd} analytic={an}"
    print(f"DenseLSTM BPTT gradient check: PASS (worst rel err {worst:.2e}, {checked} params)")


if __name__ == "__main__":
    check_lstm()
    check_bptt()
    print("All gradient checks passed.")
