"""Comparable online trace-patterning runs; no ground-truth training labels."""
import argparse
from collections import deque
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from envs.animal_learning import TracePatterning
from experiment.checkpoint import writer_lock
from experiments.checkpoint import (pack_ccn, restore_checkpoint, save_checkpoint,
                                    source_digest, unpack_ccn, write_json)
from experiments.models import OnlineLearner, make_model
from networks.incremental import IncrementalNetworks

METRICS = ['step', 'window_count', 'mse', 'zero_mse', 'mean_baseline_mse', 'mse_over_zero',
           'prediction_mean', 'target_mean', 'td_rms', 'outside_fraction',
           'valid_wait_mse', 'invalid_wait_mse', 'valid_prediction', 'invalid_prediction',
           'valid_count', 'invalid_count', 'clip_fraction', 'parameter_norm', 'trace_norm',
           'gradient_norm', 'state_norm', 'forget_mean', 'forget_saturated',
           'parameter_change', 'active_stage', 'steps_per_second', 'elapsed_seconds']
PREDICTIONS = ['step', 'cue', 'age', 'phase', 'reward', 'prediction', 'target', 'td_error']


class CCNLearner:
    def __init__(self, config):
        self.config = config
        with contextlib.redirect_stdout(io.StringIO()):
            self.network = IncrementalNetworks(
                config['lr'], config['seed'], 12, 1, config['hidden'],
                config['features_per_stage'], config['std_cap'], config['normalization_rate'])
        self.old_pred = 0.0
        self.layer = 0
        self.grad_norm = 0.0
        self.clipped = False

    def initialize(self, x):
        self.old_pred = self.network.forward(x)

    def step(self, x, reward):
        self.network.decay_gradient(self.config['gamma'] * self.config['lambda'])
        self.network.backward(self.layer)
        prediction = self.network.forward(x)
        delta = reward + self.config['gamma'] * prediction - self.old_pred
        self.network.update_parameters(self.layer, delta)
        self.old_pred = prediction
        return prediction, delta

    def parameters(self):
        values = [self.network.bias, *self.network.prediction_weights]
        for cell in self.network.LSTM_neurons:
            for gate in ('i', 'f', 'g', 'o'):
                values.extend(getattr(cell, 'w_' + gate))
                values.extend((getattr(cell, 'u_' + gate), getattr(cell, 'b_' + gate)))
        return np.asarray(values)

    def diagnostics(self):
        cells = self.network.LSTM_neurons
        forget = np.array([c.f for c in cells])
        return {'parameter_norm': float(np.linalg.norm(self.parameters())),
                'trace_norm': float(np.linalg.norm(self.network.prediction_weights_gradient)),
                'gradient_norm': None, 'state_norm': float(np.linalg.norm(self.network.get_state())),
                'forget_mean': float(forget.mean()),
                'forget_saturated': float(np.mean((forget < .05) | (forget > .95)))}


def resolve_config(preset, args):
    document = json.loads((ROOT / 'experiments/configs.json').read_text(encoding='utf-8-sig'))
    if preset not in document['presets']:
        raise ValueError(f'Unknown preset: {preset}')
    config = {**document['defaults'], **document['presets'][preset]}
    for key in ('steps', 'seed', 'lr', 'hidden', 'context', 'memory', 'layers', 'heads',
                'checkpoint_every', 'log_every', 'window', 'steps_per_stage',
                'features_per_stage', 'normalization_rate', 'clip'):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if getattr(args, 'isi', None):
        config['isi'] = args.isi
    if getattr(args, 'iti', None):
        config['iti'] = args.iti
    for key in ('steps', 'hidden', 'context', 'memory', 'layers', 'heads', 'checkpoint_every',
                'log_every', 'window', 'steps_per_stage', 'features_per_stage', 'diagnostic_every'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('lr', 'clip'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not all(math.isfinite(config[k]) and 0 <= config[k] <= 1 for k in ('gamma', 'lambda')):
        raise ValueError('gamma and lambda must be between zero and one')
    if not math.isfinite(config['std_cap']) or config['std_cap'] <= 0:
        raise ValueError('std_cap must be finite and positive')
    if not 0 <= config['normalization_rate'] <= 1 or config['seed'] < 0:
        raise ValueError('Invalid normalization rate or seed')
    for key in ('isi', 'iti'):
        low, high = config[key]
        if not 1 <= low <= high:
            raise ValueError(f'Invalid {key} range')
    if config['model'] in ('transformer', 'gtrxl') and config['hidden'] % config['heads']:
        raise ValueError('hidden must be divisible by heads')
    if config['model'] == 'ccn' and config['features_per_stage'] > config['hidden']:
        raise ValueError('features_per_stage exceeds hidden')
    return config


def parameter_vector(learner):
    if isinstance(learner, CCNLearner):
        return learner.parameters()
    with torch.no_grad():
        return torch.cat([p.flatten() for p in learner.parameters]).cpu().numpy().copy()


def env_snapshot(env):
    return {key: value for key, value in vars(env).items() if key != 'mt'}


def restore_env(state, rng):
    env = TracePatterning.__new__(TracePatterning)
    vars(env).update(state)
    env.mt = np.random.default_rng()
    env.mt.bit_generator.state = rng
    return env


def metrics_row(step, rows, diagnostics, parameter_change, layer, elapsed, rate):
    values = np.asarray(rows, dtype=float)
    means = values.mean(axis=0)
    valid, invalid = values[:, 7].sum(), values[:, 9].sum()
    result = dict(zip(METRICS, [None] * len(METRICS)))
    result.update(step=step, window_count=len(rows), mse=means[0], zero_mse=means[1],
                  mean_baseline_mse=means[2], mse_over_zero=means[0] / max(means[1], 1e-12),
                  prediction_mean=means[3], target_mean=means[4], td_rms=math.sqrt(means[5]),
                  outside_fraction=means[6], valid_count=int(valid), invalid_count=int(invalid),
                  valid_wait_mse=values[:, 8].sum() / valid if valid else None,
                  invalid_wait_mse=values[:, 10].sum() / invalid if invalid else None,
                  valid_prediction=values[:, 11].sum() / valid if valid else None,
                  invalid_prediction=values[:, 12].sum() / invalid if invalid else None,
                  clip_fraction=means[13], parameter_change=parameter_change, active_stage=layer,
                  elapsed_seconds=elapsed, steps_per_second=rate, **diagnostics)
    return result


def choose_device(requested):
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install uv sync --extra experiments or use --device cpu')
    return requested


def run_one(config, name, device='cpu', resume=None, stop_after=None):
    if not name or name in ('.', '..') or any(c in name for c in '/\\:<>"|?*') or name.endswith(('.', ' ')):
        raise ValueError('Use a single valid directory name for --name')
    directory = ROOT / 'results' / name
    with writer_lock(directory):
        return train(config, directory, device, resume, stop_after)


def train(config, directory, device, resume, stop_after):
    if not resume and any((directory / name).exists() for name in ('run.json', 'metrics.csv', 'predictions.csv')):
        raise ValueError('Run already exists: choose another --name or --resume')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(1)
    # CUDA attention/RNN operations need not be bit-identical to CPU operations.
    if device == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
        torch.cuda.reset_peak_memory_stats()
    learner = CCNLearner(config) if config['model'] == 'ccn' else OnlineLearner(
        make_model(config), config, device)
    parameters = parameter_vector(learner)
    if resume:
        state, manifest = restore_checkpoint(resume, directory, config, device)
        env = restore_env(state['env'], state['env_rng'])
        if config['model'] == 'ccn':
            learner.network = unpack_ccn(state['ccn_npz'], directory)
            learner.old_pred = state['old_pred']
        else:
            learner.restore(state['learner'])
        rows = deque(state['rows'], maxlen=config['window'])
    else:
        env = TracePatterning(tuple(config['isi']), tuple(config['iti']), 5, config['seed'])
        learner.initialize(list(env.reset()))
        rows = deque(maxlen=config['window'])
        state = {'next_step': 0, 'elapsed': 0.0, 'target_sum': 0.0, 'cue': '', 'age': 0,
                 'stream_digest': bytes(32), 'last_parameters': parameters.tolist()}
        state['cue'] = ''.join(str(int(v)) for v in env.current_state[:6])
        manifest = {'config': config, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'device': 'cpu' if config['model'] == 'ccn' else device,
                    'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                    'python': platform.python_version(), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'parameter_count': len(parameters),
                    'source_sha256': source_digest(), 'checkpoints': [],
                    'protocol': 'online TD(lambda), one transition and update per step; no target labels'}
    initial_step = state['next_step']
    previous_elapsed = state['elapsed']
    started = time.perf_counter()
    log_started = started
    log_step = initial_step
    diagnostic = state.get('diagnostic', learner.diagnostics())
    with contextlib.ExitStack() as stack:
        streams = {name: stack.enter_context((directory / name).open(
            'a' if resume else 'w', newline='', encoding='utf-8'))
                   for name in ('metrics.csv', 'predictions.csv')}
        writers = {name: csv.DictWriter(streams[name], fieldnames=columns)
                   for name, columns in (('metrics.csv', METRICS), ('predictions.csv', PREDICTIONS))}
        if not resume:
            for writer in writers.values():
                writer.writeheader()

        def snapshot(status):
            for stream in streams.values():
                stream.flush()
                os.fsync(stream.fileno())
            state.update(elapsed=previous_elapsed + time.perf_counter() - started,
                         env=env_snapshot(env), env_rng=env.mt.bit_generator.state,
                         rows=list(rows), old_pred=learner.old_pred, diagnostic=diagnostic)
            if config['model'] == 'ccn':
                state['ccn_npz'] = pack_ccn(learner.network, directory)
            else:
                state['learner'] = learner.snapshot()
            return save_checkpoint(directory, state, config, manifest, status)

        if not resume:
            snapshot('running')
        else:
            manifest['status'] = 'running'
            write_json(directory / 'run.json', manifest)
        print(f"{directory.name}: {config['model']} {manifest['device']}, {len(parameters)} parameters", flush=True)
        try:
            for i in range(initial_step, config['steps']):
                x = list(env.step())
                reward = env.get_US()
                if sum(x[:6]) == 3:
                    state['cue'] = ''.join(str(int(v)) for v in x[:6])
                    state['age'] = 0
                else:
                    state['age'] += 1
                waiting = env.remaining_until_US > 0
                valid, invalid = waiting and env.valid, waiting and not env.valid
                phase = 'valid_wait' if valid else 'invalid_wait' if invalid else 'quiet'
                layer = min((i + 1) // config['steps_per_stage'],
                            (config['hidden'] - 1) // config['features_per_stage'])
                if config['model'] == 'ccn':
                    learner.layer = layer
                prediction, delta = learner.step(x, reward)
                target = env.get_target(config['gamma'])
                if not all(math.isfinite(value) for value in (prediction, delta, target)):
                    raise FloatingPointError('Non-finite prediction or TD error')
                error = (prediction - target) ** 2
                baseline = state['target_sum'] / i if i else 0.0
                rows.append([error, target * target, (baseline - target) ** 2, prediction,
                             target, delta * delta, float(prediction < 0 or prediction > 1),
                             float(valid), error * valid, float(invalid), error * invalid,
                             prediction * valid, prediction * invalid, float(learner.clipped)])
                state['target_sum'] += target
                # Audit that every architecture received exactly the same generated data.
                state['stream_digest'] = hashlib.sha256(state['stream_digest'] +
                    np.asarray(x + [reward, target], dtype='<f8').tobytes()).digest()
                state['next_step'] = i + 1
                if i % 10 == 0 or sum(x[:6]) == 3 or reward:
                    writers['predictions.csv'].writerow(dict(zip(PREDICTIONS,
                        (i + 1, state['cue'], state['age'], phase, reward, prediction, target, delta))))
                if (i + 1) % config['diagnostic_every'] == 0:
                    diagnostic = learner.diagnostics()
                if (i + 1) % config['log_every'] == 0 or i + 1 == config['steps']:
                    now = time.perf_counter()
                    parameters = parameter_vector(learner)
                    change = float(np.linalg.norm(parameters - np.array(state['last_parameters'])))
                    state['last_parameters'] = parameters.tolist()
                    row = metrics_row(i + 1, rows, diagnostic, change,
                        layer if config['model'] == 'ccn' else 0,
                        previous_elapsed + now - started, (i + 1 - log_step) / (now - log_started))
                    writers['metrics.csv'].writerow(row)
                    for stream in streams.values():
                        stream.flush()
                    print(f"  step={i+1} mse={row['mse']:.6f} zero={row['zero_mse']:.6f} "
                          f"ratio={row['mse_over_zero']:.3f} rate={row['steps_per_second']:.0f}/s", flush=True)
                    log_step, log_started = i + 1, now
                if stop_after and i + 1 >= stop_after:
                    snapshot('paused_for_test')
                    return manifest
                if (i + 1) % config['checkpoint_every'] == 0 and i + 1 < config['steps']:
                    snapshot('running')
            if device == 'cuda':
                manifest['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
            manifest['stream_sha256'] = state['stream_digest'].hex()
            snapshot('completed')
        except BaseException as exc:
            # Resume uses the last complete snapshot, not a potentially half-updated step.
            manifest.update(status='interrupted_or_failed', failure=f'{type(exc).__name__}: {exc}',
                            last_attempted_step=state['next_step'])
            write_json(directory / 'run.json', manifest)
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', default='lstm', help='rnn, gru, lstm, transformer, gtrxl, ccn, columnar, all')
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Save a complete snapshot and stop at this step')
    for key in ('steps', 'seed', 'hidden', 'context', 'memory', 'layers', 'heads', 'checkpoint-every',
                'log-every', 'window', 'steps-per-stage', 'features-per-stage'):
        parser.add_argument('--' + key, type=int)
    for key in ('lr', 'normalization-rate', 'clip'):
        parser.add_argument('--' + key, type=float)
    parser.add_argument('--isi', type=int, nargs=2)
    parser.add_argument('--iti', type=int, nargs=2)
    args = parser.parse_args()
    if args.preset == 'all' and args.resume:
        parser.error('Resume a single preset, not all')
    try:
        device = choose_device(args.device)
        presets = ('rnn', 'gru', 'lstm', 'transformer', 'gtrxl', 'columnar', 'ccn') if args.preset == 'all' else (args.preset,)
        for preset in presets:
            config = resolve_config(preset, args)
            name = f'{args.name}_{preset}' if args.preset == 'all' else args.name
            run_one(config, name, device, args.resume, args.stop_after)
    except (ValueError, OSError, FloatingPointError, RuntimeError) as exc:
        print(f'Experiment failed: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
