"""Finite-difference verification of the ported gradient computations.

Checks that the RTRL traces in nn/lstm.py produce gradients matching
central finite differences of the replayed hidden state, and that the
truncated-BPTT gradients in networks/bptt.py match finite differences of
the replayed windowed prediction sum.
"""
import argparse
import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nn.neuron import LinearNeuron
from nn.lstm import LSTM, LSTMNormalied
from networks.bptt import DenseLSTM, DenseLSTMRmsProp
from networks.incremental import IncrementalNetworks, NormalizedIncrementalNetworks, Snap1
from envs.animal_learning import TracePatterning
from experiment.checkpoint import save_state, load_state


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

    def __init__(self, seed=0, n_inputs=3, T=6, normalized=False):
        rng = np.random.default_rng(seed)
        self.T = T
        self.inputs = rng.integers(0, 2, size=(T, n_inputs)).astype(float)
        self.input_neurons = [LinearNeuron(True, False) for _ in range(n_inputs)]

        def r():
            return float(rng.uniform(-0.01, 0.01))

        cls = LSTMNormalied if normalized else LSTM
        self.cell = cls(r(), r(), r(), r(), r(), r(), r(), r(),
                        std_cap=0.1, decay_rate=1.0 if normalized else 0.999)
        if normalized:
            # Hold statistics constant: the algorithm does not differentiate
            # through the running normalization statistics.
            self.cell.state_mean = .2
            self.cell.state_variance = 1.7
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
        cell.zero_grad()
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


def check_lstm(normalized=False):
    probe = CellProbe(normalized=normalized)
    analytic = probe.analytic_grad()
    worst = 0.0
    for name in probe.names:
        fd = central_diff(probe, probe.params[name])
        an = analytic[name]
        scale = max(1.0, abs(fd), abs(an))
        rel = abs(fd - an) / scale
        worst = max(worst, rel)
        assert rel < 1e-7, f"LSTM grad mismatch at {name}: fd={fd} analytic={an}"
    print(f"{'Normalized ' if normalized else ''}LSTM RTRL gradient check: PASS (max scaled error {worst:.2e}, "
          f"{len(probe.names)} params)")


class BPTTProbe:
    def __init__(self, seed=0, input_size=3, hidden=4, trunc=5, T=7):
        self.net = DenseLSTM(step_size=0.01, seed=seed, hidden_size=hidden,
                             no_of_input_features=input_size, truncation=trunc)
        rng = np.random.default_rng(seed + 1)
        self.inputs = rng.integers(0, 2, size=(T, input_size)).astype(float)
        self.T = T
        self.net.prediction_weights[:] = rng.uniform(-0.5, 0.5, hidden)
        self.boundaries = self.reference_states()

    def reference_step(self, x, h, c):
        # Independent forward calculation; no ring-buffer state is used.
        z = np.array([self.net.W[g] @ x + self.net.U[g] @ h + self.net.b[g]
                      for g in range(4)])
        i, f, o = [1 / (1 + np.exp(-z[g])) for g in (0, 1, 3)]
        c = f * c + i * np.tanh(z[2])
        return o * np.tanh(c), c

    def reference_states(self):
        states = [(np.zeros(self.net.hidden_state_size),
                   np.zeros(self.net.hidden_state_size))]
        for x in self.inputs:
            states.append(self.reference_step(x, *states[-1]))
        return states

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
        total = 0.0
        for end in range(1, self.T + 1):
            start = max(0, end - self.net.truncation)
            # Truncated BPTT treats the window's initial state as constant.
            h, c = (a.copy() for a in self.boundaries[start])
            for x in self.inputs[start:end]:
                h, c = self.reference_step(x, h, c)
            total += h @ self.net.prediction_weights
        return float(total)

    @property
    def params(self):
        net = self.net
        return {"W": net.W, "U": net.U, "b": net.b, "pred": net.prediction_weights}


def check_bptt(T=7, trunc=5):
    probe = BPTTProbe(T=T, trunc=trunc)
    probe.run_backward()
    net = probe.net
    targets = [("W", net.W_grad), ("U", net.U_grad), ("b", net.b_grad),
               ("pred", net.prediction_weights_grad)]
    worst = 0.0
    checked = 0
    for name, grad_arr in targets:
        flat_grad = np.asarray(grad_arr).ravel().copy()
        idxs = range(len(flat_grad))
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
            assert rel < 1e-7, f"BPTT grad mismatch {name}[{idx}]: fd={fd} analytic={an}"
    probe._fresh()
    for t, x in enumerate(probe.inputs, 1):
        probe.net.forward(x)
        np.testing.assert_allclose(probe.net.get_state(),
                                   np.concatenate(probe.boundaries[t]), atol=1e-12)
    print(f"BPTT T={T}, K={trunc}: PASS (max scaled error {worst:.2e}, {checked} params)")


def check_updates():
    net = DenseLSTMRmsProp(.03, 2, 3, 2, 2, .9, 1e-8)
    rng = np.random.default_rng(4)
    for param, grad, avg in ((net.W, net.W_grad, net.W_grad_rmsprop),
                             (net.U, net.U_grad, net.U_grad_rmsprop),
                             (net.b, net.b_grad, net.b_grad_rmsprop),
                             (net.prediction_weights, net.prediction_weights_grad,
                              net.prediction_weights_grad_rmsprop)):
        grad[:] = rng.normal(size=grad.shape)
    for error in (.7, -.2):
        expected = []
        for param, grad, avg in ((net.W, net.W_grad, net.W_grad_rmsprop),
                                 (net.U, net.U_grad, net.U_grad_rmsprop),
                                 (net.b, net.b_grad, net.b_grad_rmsprop),
                                 (net.prediction_weights, net.prediction_weights_grad,
                                  net.prediction_weights_grad_rmsprop)):
            squared = .9 * avg + .1 * (grad * error) ** 2
            expected.append((param.copy() + .03 * error * grad / (np.sqrt(squared) + 1e-8), squared))
        net.update_parameters(0, error)
        for (want, squared), param, avg in zip(expected,
                (net.W, net.U, net.b, net.prediction_weights),
                (net.W_grad_rmsprop, net.U_grad_rmsprop, net.b_grad_rmsprop,
                 net.prediction_weights_grad_rmsprop)):
            np.testing.assert_allclose(param, want)
            np.testing.assert_allclose(avg, squared)
    grads = [net.W_grad, net.U_grad, net.b_grad, net.prediction_weights_grad]
    saved = [g.copy() for g in grads]
    net.decay_gradient(.6)
    for g, old in zip(grads, saved):
        np.testing.assert_allclose(g, old * .6)
    net.zero_grad()
    assert all(np.count_nonzero(g) == 0 for g in grads)
    print("Dense RMSProp, gradient decay and zeroing: PASS")


def check_cells_and_freezing():
    for cls in (IncrementalNetworks, NormalizedIncrementalNetworks, Snap1):
        args = (.01, 0, 2, 1, 4, 2, .1)
        net = cls(*args) if cls is Snap1 else cls(*args, .9)
        net.prediction_weights[:] = [.2, -.1, .3, .1]
        for _ in range(3):
            net.forward([1., 0.])
            net.backward(0)
        cells = net.LSTM_neurons
        frozen = [(c.w_i.copy(), c.u_i, c.b_i, c.input_means.copy()) for c in cells[:2]]
        active = [c.w_i.copy() for c in cells[2:]]
        net.backward(1)
        net.update_parameters(1, .5)
        net.forward([0., 1.])
        net.backward(1)
        if cls is not Snap1:
            for c, (w, u, b, means) in zip(cells[:2], frozen):
                np.testing.assert_array_equal(c.w_i, w)
                assert c.u_i == u and c.b_i == b
                np.testing.assert_array_equal(c.input_means, means)
                assert not c.update_statistics_flag
        assert any(np.any(c.w_i != w) for c, w in zip(cells[2:], active))
        for c in cells:
            saved = c.Gw_i.copy()
            trace = c.Hw_i.copy()
            c.decay_gradient(.4)
            np.testing.assert_allclose(c.Gw_i, .4 * saved)
            np.testing.assert_array_equal(c.Hw_i, trace)
            c.zero_grad()
            assert not np.any(c.Gw_i)
        net.reset_state()
        assert all(c.h == 0 and c.c == 0 and not np.any(c.Hw_i) for c in cells)
    cell = CellProbe().cell
    cell.input_means[:] = [.1, .2, .3]
    cell.input_std[:] = [1., 2., 3.]
    for n, value in zip(cell.incoming_neurons, [1., 0., 2.]):
        n.value = value
    means = cell.input_means * .999 + .001 * np.array([1., 0., 2.])
    variance = np.maximum(cell.input_std * .999 + .001 * (np.array([1., 0., 2.]) - means) ** 2, .1)
    cell.update_statistics()
    np.testing.assert_allclose(cell.input_means, means)
    np.testing.assert_allclose(cell.input_std, variance)
    cell.set_update_statistics(False)
    cell.update_statistics()
    np.testing.assert_array_equal(cell.input_means, means)
    # Exercise every gate's input, recurrent and bias RMSProp accumulator.
    expected = []
    for gate in ("i", "f", "g", "o"):
        for param, grad, avg in (("w_", "Gw_", "beta_2_w_"),
                                 ("u_", "Gu_", "beta_2_u_"),
                                 ("b_", "Gb_", "beta_2_b_")):
            name, gname, aname = param + gate, grad + gate, avg + gate
            g = np.ones_like(getattr(cell, gname)) * .3
            setattr(cell, gname, g)
            squared = .9999 * getattr(cell, aname) + .0001 * (g * -.4) ** 2
            want = getattr(cell, name) + .02 * g * -.4 / (np.sqrt(squared) + 1e-8)
            expected.append((name, aname, want, squared))
    cell.update_weights(.02, -.4)
    for name, aname, want, squared in expected:
        np.testing.assert_allclose(getattr(cell, name), want)
        np.testing.assert_allclose(getattr(cell, aname), squared)
    print("Cell normalization, gradient traces and layer freezing: PASS")


def check_checkpoint_roundtrip():
    with contextlib.redirect_stdout(io.StringIO()):
        nets = [IncrementalNetworks(.001,0,12,1,4,2,.1,.999),
                NormalizedIncrementalNetworks(.001,0,12,1,4,2,.1,.999),
                Snap1(.001,0,12,1,4,2,.1), DenseLSTMRmsProp(.001,0,3,12,2,.999,1e-8)]
    with tempfile.TemporaryDirectory(prefix='ccn-roundtrip-') as directory:
        for net in nets:
            env = TracePatterning((24,36),(80,120),5,0)
            old = net.forward(env.reset())
            def step(n, e, previous):
                x = e.step()
                n.decay_gradient(.891)
                n.backward(0)
                pred = n.forward(x)
                n.update_parameters(0, e.get_US() + .9 * pred - previous)
                return pred
            for _ in range(30):
                old = step(net, env, old)
            path = Path(directory) / 'state.npz'
            save_state(path, {'network': net, 'env': env, 'old_pred': old}, {'test': True})
            state, metadata = load_state(path)
            assert metadata == {'test': True}
            restored, restored_env, restored_old = state['network'], state['env'], state['old_pred']
            for _ in range(200):
                old = step(net, env, old)
                restored_old = step(restored, restored_env, restored_old)
                assert old == restored_old
                np.testing.assert_array_equal(net.prediction_weights, restored.prediction_weights)
                assert env.mt.bit_generator.state == restored_env.mt.bit_generator.state
            print(type(net).__name__ + ' disk checkpoint replay: exact')


def check_cpp(path):
    output = subprocess.check_output([str(Path(path).resolve())], text=True)
    actual = {'cell': [], 'dense': []}
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] in actual:
            actual[fields[0]].append(np.array([float(v) for v in fields[1:]]))
    cell = LSTM(.02,.03,-.01,.04,.01,-.02,.03,.02,.1,.999)
    inputs = [LinearNeuron(True,False) for _ in range(3)]
    for j, neuron in enumerate(inputs):
        cell.add_synapse(neuron,.01*(j+1),-.02*(j+1),.03*(j+1),-.01*(j+1))
    expected = {'cell': [], 'dense': []}
    for t in range(8):
        for j, neuron in enumerate(inputs):
            neuron.value = (t+j)%2
        cell.update_value_sync(); cell.fire(); cell.decay_gradient(.891)
        cell.compute_gradient_of_all_synapses(); cell.accumulate_gradient(.3)
        cell.update_weights(.001,-.2)
        arrays = [getattr(cell, prefix+gate) for prefix in ('w_', 'Gw_') for gate in ('i','f','g','o')]
        scalars = [getattr(cell, prefix+gate) for prefix in ('Gu_', 'Gb_', 'u_', 'b_') for gate in ('i','f','g','o')]
        expected['cell'].append(np.concatenate(([cell.h,cell.c], *arrays, scalars)))
    net = DenseLSTMRmsProp(.001,0,2,3,8,.9999,1e-8)
    for array, modulus in ((net.W,7),(net.U,5),(net.b,3)):
        array.flat[:] = .01 * (np.arange(array.size)%modulus - modulus//2)
    net.prediction_weights[:] = [.2,-.1]
    for t in range(3):
        prediction = net.forward([t%2,(t+1)%2,t%2]); net.decay_gradient(.891)
        net.backward(0); net.update_parameters(0,-.2)
        expected['dense'].append(np.concatenate([[prediction],net.get_state(), net.W.ravel(),net.U.ravel(),
            net.b.ravel(),net.prediction_weights,net.W_grad.ravel(),net.U_grad.ravel(),net.b_grad.ravel(),net.prediction_weights_grad]))
    for kind in actual:
        assert len(actual[kind]) == len(expected[kind]), 'Missing C++ probe output'
        a, b = np.array(actual[kind]), np.array(expected[kind])
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-5)
        print(f'C++ {kind} predictions/state/gradients/updates: PASS, max absolute error {np.max(np.abs(a-b)):.2e}')


def check_resume_cli():
    import sys
    root = Path(__file__).resolve().parent
    params = {'seed':['0'], 'steps':['450'], 'normalization_rate':['.999'],
              'input_features':['12'], 'features':['4:2:200'], 'step_size':['.001'],
              'start':['0'], 'lambda':['.99'], 'std_cap':['.1']}
    with tempfile.TemporaryDirectory(prefix='ccn-cli-resume-') as directory:
        config = Path(directory) / 'config.json'
        for algorithm in ('hybrid', 'tbptt_rmsprop'):
            params['algorithm'] = [{algorithm: {} if algorithm == 'hybrid' else
                {'features_truncation':['3:2'], 'beta_2':['.9999'], 'eps':['1e-8']}}]
            config.write_text(json.dumps({'experiment':{'name':algorithm, 'params':params}}), encoding='utf-8')
            command = [sys.executable, '-B', str(root/'run_animal_learning.py'),
                       '--config', str(config), '--run', '0', '--checkpoint-every', '100']
            def execute(extra=()):
                result = subprocess.run(command + list(extra), cwd=directory,
                                        capture_output=True, text=True, timeout=60)
                assert result.returncode == 0, result.stderr
            execute()
            out = Path(directory) / 'results' / algorithm
            paths = [out / name for name in ('runs.csv','error_table.csv','predictions.csv')]
            original = [path.read_bytes() for path in paths]
            state, _ = load_state(out/'checkpoint_0_450.npz')
            execute(('--resume',str(out/'checkpoint_0_100.npz')))
            assert original == [path.read_bytes() for path in paths]
            restored, _ = load_state(out/'checkpoint_0_450.npz')
            assert state['old_pred'] == restored['old_pred']
            assert state['running_error'] == restored['running_error']
            np.testing.assert_array_equal(state['network'].prediction_weights,
                                          restored['network'].prediction_weights)
            print(algorithm + ' CLI resume across transitions: exact CSV and weights')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpp', help='Path to the compiled verify_cpp probe')
    args = parser.parse_args()
    check_lstm()
    check_lstm(normalized=True)
    for T, trunc in ((3, 8), (5, 5), (7, 5), (16, 3), (6, 1)):
        check_bptt(T, trunc)
    check_updates()
    check_cells_and_freezing()
    check_checkpoint_roundtrip()
    check_resume_cli()
    if args.cpp:
        check_cpp(args.cpp)
    print("All gradient checks passed.")
