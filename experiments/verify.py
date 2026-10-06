"""Behavioral checks for causal prediction, TD updates and disk recovery."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from envs.animal_learning import TracePatterning
from experiments.checkpoint import restore_checkpoint
from experiments.gtrxl import GRUGate
from experiments.models import OnlineLearner, make_model
from experiments.run import resolve_config, run_one


def configuration(kind, **overrides):
    config = resolve_config(kind, argparse.Namespace())
    config.update(hidden=8, context=11, memory=12, layers=1, steps=180,
                  log_every=37, checkpoint_every=60, window=100,
                  features_per_stage=2, steps_per_stage=60, diagnostic_every=19)
    config.update(overrides)
    return config


def check_causality():
    torch.manual_seed(9)
    for kind in ('transformer', 'gtrxl'):
        model = make_model(configuration(kind, layers=2))
        torch.nn.init.normal_(model.head.weight)
        x = torch.randn(1, 11, 12)
        changed = x.clone()
        changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 10
        state = model.initial_state('cpu')
        first, _ = model(x, state)
        second, _ = model(changed, state)
        torch.testing.assert_close(first[:, :6], second[:, :6], rtol=0, atol=1e-6)
        assert not torch.allclose(first[:, 6:], second[:, 6:]), 'Test must change later outputs'
    print('PASS: future inputs cannot change past Transformer/GTrXL predictions')


def check_segment_memory():
    torch.manual_seed(3)
    model = make_model(configuration('gtrxl', layers=2, memory=4))
    torch.nn.init.normal_(model.head.weight)
    x = torch.randn(1, 13, 12)
    with torch.no_grad():
        whole, _ = model(x)
        state = model.initial_state('cpu')
        pieces = []
        for i in range(x.shape[1]):
            pred, state = model(x[:, i:i+1], state)
            pieces.append(pred)
            assert all(memory.shape[1] <= 4 and not memory.requires_grad for memory in state)
        torch.testing.assert_close(whole, torch.cat(pieces, dim=1), rtol=1e-5, atol=2e-6)
    gate = GRUGate(8)
    with torch.no_grad():
        gate.bias.fill_(30)
        source, branch = torch.randn(2, 8), torch.randn(2, 8)
        torch.testing.assert_close(gate(source, branch), source, rtol=0, atol=1e-6)
    print('PASS: GTrXL streaming/segment agreement, bounded detached memory and identity gate')


def check_td_update():
    config = configuration('lstm', context=4, clip=1e10)
    learner = OnlineLearner(make_model(config), config, 'cpu')
    learner.initialize([1.0] + [0.0] * 11)
    old = learner.old_pred
    prediction, delta = learner.step([0.0] * 12, 1.0)
    assert abs(delta - (1.0 + config['gamma'] * prediction - old)) < 1e-7
    assert learner.model.head.bias.item() > 0, 'Positive TD error must increase value'
    prior = [value.clone() for value in learner.traces]
    previous, _ = learner.model(learner._window(), learner.boundaries[0])
    gradient = torch.autograd.grad(previous[0, -1], learner.parameters)
    learner.step([0.0] * 12, 0.0)
    for trace, before, increment in zip(learner.traces, prior, gradient):
        torch.testing.assert_close(trace, config['gamma'] * config['lambda'] * before + increment)
    for _ in range(8):
        learner.step([0.0] * 12, 0.0)
    assert len(learner.observations) == 4 and len(learner.boundaries) == 4
    assert learner.state[0].norm() > 0, 'Memory must persist across window boundaries'
    print('PASS: TD sign, eligibility decay, bounded gradient history and persistent LSTM state')


def check_delayed_credit():
    model = make_model(configuration('rnn', hidden=4, context=48))
    with torch.no_grad():
        model.core.weight_ih_l0.fill_(.05)
        model.core.weight_hh_l0.copy_(torch.eye(4) * .95)
        model.core.bias_ih_l0.zero_()
        model.core.bias_hh_l0.zero_()
        model.head.weight.fill_(1)
    x = torch.zeros(1, 40, 12, requires_grad=True)
    output, _ = model(x)
    full_gradient = torch.autograd.grad(output[0, -1], x)[0]
    assert full_gradient[0, 0].norm() > 0
    with torch.no_grad():
        _, boundary = model(x[:, :32])
    shortened = x.detach().clone().requires_grad_()
    output, _ = model(shortened[:, 32:], boundary)
    short_gradient = torch.autograd.grad(output[0, -1], shortened)[0]
    assert short_gradient[0, 0].norm() == 0
    assert short_gradient[0, -1].norm() > 0
    print('PASS: a short BPTT window removes the direct early-cue gradient path')


def metrics_without_time(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return [{key: value for key, value in row.items()
                 if key not in ('elapsed_seconds', 'steps_per_second')}
                for row in csv.DictReader(stream)]


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree_equal(left, right)
    else:
        assert first == second


def check_recovery():
    prefix = 'verify_experiments_' + uuid.uuid4().hex[:8]
    streams = set()
    for kind in ('rnn', 'gru', 'lstm', 'transformer', 'gtrxl', 'ccn'):
        config = configuration(kind)
        full_name, split_name = f'{prefix}_{kind}_full', f'{prefix}_{kind}_split'
        with contextlib.redirect_stdout(io.StringIO()):
            full = run_one(config, full_name)
            run_one(config, split_name, stop_after=73)
        full_dir, split_dir = ROOT / 'results' / full_name, ROOT / 'results' / split_name
        checkpoint = split_dir / 'checkpoint_000000073.pt'
        checkpoint_content = checkpoint.read_bytes()
        checkpoint.write_bytes(checkpoint_content[:-1] + bytes([checkpoint_content[-1] ^ 1]))
        try:
            restore_checkpoint(checkpoint, split_dir, config, 'cpu')
        except ValueError as exc:
            assert 'checksum' in str(exc)
        else:
            raise AssertionError('Changed checkpoint was accepted')
        checkpoint.write_bytes(checkpoint_content)
        prefix_content = (split_dir / 'metrics.csv').read_bytes()
        # Reject changed history before truncating any other log.
        (split_dir / 'metrics.csv').write_bytes(b'X' + prefix_content[1:])
        try:
            restore_checkpoint(checkpoint, split_dir, config, 'cpu')
        except ValueError as exc:
            assert 'prefix' in str(exc)
        else:
            raise AssertionError('Changed metric prefix was accepted')
        (split_dir / 'metrics.csv').write_bytes(prefix_content)
        # Simulate records written after the last complete snapshot.
        with (split_dir / 'predictions.csv').open('a', encoding='utf-8') as stream:
            stream.write('uncommitted tail\n')
        with contextlib.redirect_stdout(io.StringIO()):
            split = run_one(config, split_name, resume=checkpoint)
        assert full['stream_sha256'] == split['stream_sha256']
        streams.add(full['stream_sha256'])
        assert (full_dir / 'predictions.csv').read_bytes() == (split_dir / 'predictions.csv').read_bytes()
        assert metrics_without_time(full_dir / 'metrics.csv') == metrics_without_time(split_dir / 'metrics.csv')
        left = torch.load(full_dir / 'checkpoint_000000180.pt', weights_only=True)
        right = torch.load(split_dir / 'checkpoint_000000180.pt', weights_only=True)
        for key in ('learner', 'old_pred', 'env', 'env_rng', 'rows', 'last_parameters'):
            if key in left:
                assert_tree_equal(left[key], right[key])
        if kind == 'ccn':
            from experiments.checkpoint import unpack_ccn
            from experiments.run import CCNLearner
            a, b = CCNLearner(config), CCNLearner(config)
            a.network = unpack_ccn(left['ccn_npz'], full_dir)
            b.network = unpack_ccn(right['ccn_npz'], split_dir)
            np.testing.assert_array_equal(a.parameters(), b.parameters())
        print(f'PASS: {kind} disk recovery, exact metrics/weights and CSV rollback')
    assert len(streams) == 1, 'Architectures received different streams'
    print('PASS: all architectures received identical observations, rewards and evaluation targets')


def check_cuda():
    if not torch.cuda.is_available():
        raise AssertionError('CUDA requested but unavailable')
    for kind in ('lstm', 'transformer', 'gtrxl'):
        config = configuration(kind)
        learner = OnlineLearner(make_model(config), config, 'cuda')
        learner.initialize([0.0] * 12)
        for i in range(5):
            pred, error = learner.step([float(i % 2)] + [0.0] * 11, float(i == 3))
            assert np.isfinite(pred) and np.isfinite(error)
        assert all(parameter.is_cuda for parameter in learner.parameters)
    print(f'PASS: CUDA forward/backward/updates on {torch.cuda.get_device_name()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_causality()
    check_segment_memory()
    check_td_update()
    check_delayed_credit()
    check_recovery()
    if args.cuda:
        check_cuda()
    print('All experiment checks passed')


if __name__ == '__main__':
    main()
