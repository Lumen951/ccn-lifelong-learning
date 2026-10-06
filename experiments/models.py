"""Small sequence predictors and one-transition TD(lambda) updates."""
from collections import deque
import math

import torch
from torch import nn


def detach_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, tuple):
        return tuple(detach_state(x) for x in state)
    if isinstance(state, list):
        return [detach_state(x) for x in state]
    return state


class RecurrentPredictor(nn.Module):
    def __init__(self, kind, hidden, inputs=12, layers=1):
        super().__init__()
        cls = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[kind]
        self.core = cls(inputs, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.kind = kind
        self.hidden = hidden
        self.layers = layers
        if kind == 'lstm':
            # Sum of the two forget-gate biases is one.
            with torch.no_grad():
                for name, value in self.core.named_parameters():
                    if 'bias' in name:
                        value[hidden:2 * hidden].fill_(0.5)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initial_state(self, device):
        h = torch.zeros(self.layers, 1, self.hidden, device=device)
        return (h, h.clone()) if self.kind == 'lstm' else h

    def forward(self, x, state=None):
        features, state = self.core(x, state)
        return self.head(features).squeeze(-1), state


class CausalTransformer(nn.Module):
    def __init__(self, hidden, heads, layers, context, inputs=12):
        super().__init__()
        self.embed = nn.Linear(inputs, hidden)
        block = nn.TransformerEncoderLayer(
            hidden, heads, 4 * hidden, dropout=0.0, batch_first=True,
            norm_first=True, activation='gelu')
        self.core = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        position = torch.arange(context).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000) / hidden))
        encoding = torch.zeros(context, hidden)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[:encoding[:, 1::2].shape[1]])
        self.register_buffer('position', encoding)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initial_state(self, device):
        return None

    def forward(self, x, state=None):
        length = x.shape[1]
        features = self.embed(x) + self.position[:length]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        features = self.norm(self.core(features, mask=mask))
        return self.head(features).squeeze(-1), None


def make_model(config):
    kind = config['model']
    if kind in ('rnn', 'gru', 'lstm'):
        return RecurrentPredictor(kind, config['hidden'], layers=config['layers'])
    if kind == 'transformer':
        return CausalTransformer(config['hidden'], config['heads'], config['layers'],
                                 config['context'])
    if kind == 'gtrxl':
        from experiments.gtrxl import GTrXL
        return GTrXL(config['hidden'], config['heads'], config['layers'], config['memory'])
    raise ValueError(f'Unknown model: {kind}')


class OnlineLearner:
    """Recompute a bounded recent window; detach all state before that window.

    Recurrent states are never reset at cue boundaries. Stored boundary states
    were produced with the parameters then in use, as in online truncated BPTT.
    GTrXL keeps detached, per-layer memory preceding the current segment.
    Only recent observations are replayed to reconstruct gradients, not losses.
    """
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.parameters = list(model.parameters())
        self.optimizer = torch.optim.RMSprop(self.parameters, lr=config['lr'],
                                            alpha=0.9999, eps=1e-8)
        # Start the squared-gradient averages at one, matching the CCN port.
        for p in self.parameters:
            self.optimizer.state[p] = {
                'step': torch.tensor(0.0), 'square_avg': torch.ones_like(p)}
        self.traces = [torch.zeros_like(p) for p in self.parameters]
        self.observations = deque()
        self.boundaries = deque()
        self.state = model.initial_state(device)
        self.old_pred = 0.0
        self.grad_norm = 0.0
        self.clipped = False

    def _append(self, x):
        self.boundaries.append(detach_state(self.state))
        self.observations.append(torch.as_tensor(x, device=self.device, dtype=torch.float32))
        if len(self.observations) > self.config['context']:
            self.observations.popleft()
            self.boundaries.popleft()

    def _window(self):
        return torch.stack(list(self.observations)).unsqueeze(0)

    def initialize(self, x):
        self._append(x)
        with torch.no_grad():
            output, self.state = self.model(self._window(), self.boundaries[0])
        self.state = detach_state(self.state)
        self.old_pred = float(output[0, -1])

    def step(self, x, reward):
        # The gradient belongs to the preceding observation, before observing x.
        previous, _ = self.model(self._window(), self.boundaries[0])
        gradients = torch.autograd.grad(previous[0, -1], self.parameters, allow_unused=True)
        with torch.no_grad():
            decay = self.config['gamma'] * self.config['lambda']
            for trace, gradient in zip(self.traces, gradients):
                trace.mul_(decay)
                if gradient is not None:
                    trace.add_(gradient)
            self._append(x)
            if self.config['model'] == 'transformer':
                output, self.state = self.model(self._window())
            else:
                # Advance persistent state only once; window replays have no side effects.
                output, self.state = self.model(self.observations[-1][None, None], self.state)
            self.state = detach_state(self.state)
            prediction = float(output[0, -1])
            delta = reward + self.config['gamma'] * prediction - self.old_pred
        self.optimizer.zero_grad(set_to_none=True)
        for p, trace in zip(self.parameters, self.traces):
            p.grad = -delta * trace
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config['clip'],
                                             error_if_nonfinite=True)
        self.grad_norm = float(norm)
        self.clipped = self.grad_norm > self.config['clip']
        self.optimizer.step()
        self.old_pred = prediction
        return prediction, delta

    def diagnostics(self):
        with torch.no_grad():
            norm = float(torch.sqrt(sum(p.square().sum() for p in self.parameters)))
            trace = float(torch.sqrt(sum(t.square().sum() for t in self.traces)))
            state = self.state
            if isinstance(state, tuple):
                state = state[0]
            if isinstance(state, list):
                state = torch.cat([x.flatten() for x in state])
            if state is None:
                state_norm = 0.0
            else:
                state_norm = float(state.norm())
            output = {'parameter_norm': norm, 'trace_norm': trace,
                      'state_norm': state_norm, 'gradient_norm': self.grad_norm}
            if self.config['model'] == 'lstm':
                x = self.observations[-1]
                h = self.boundaries[-1][0][0, 0]
                gates = (self.model.core.weight_ih_l0 @ x + self.model.core.bias_ih_l0
                         + self.model.core.weight_hh_l0 @ h + self.model.core.bias_hh_l0)
                hidden = self.config['hidden']
                forget = gates[hidden:2 * hidden].sigmoid()
                output.update(forget_mean=float(forget.mean()),
                              forget_saturated=float(((forget < .05) | (forget > .95)).float().mean()))
            return output

    def snapshot(self):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'traces': self.traces, 'observations': list(self.observations),
                'boundaries': list(self.boundaries), 'state': self.state,
                'old_pred': self.old_pred, 'grad_norm': self.grad_norm, 'clipped': self.clipped}

    def restore(self, saved):
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.traces = saved['traces']
        self.observations = deque(saved['observations'])
        self.boundaries = deque(saved['boundaries'])
        self.state = saved['state']
        self.old_pred = saved['old_pred']
        self.grad_norm = saved['grad_norm']
        self.clipped = saved['clipped']
