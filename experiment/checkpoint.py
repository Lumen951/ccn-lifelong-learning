"""Versioned, atomic training snapshots without pickle."""
import csv
import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from envs.animal_learning import TracePatterning
from networks.bptt import DenseLSTM, DenseLSTMRmsProp
from networks.incremental import IncrementalNetworks, NormalizedIncrementalNetworks, Snap1
from nn.lstm import LSTM, LSTMNormalied
from nn.neuron import LinearNeuron

_CLASSES = {c.__name__: c for c in (TracePatterning, DenseLSTM, DenseLSTMRmsProp,
            IncrementalNetworks, NormalizedIncrementalNetworks, Snap1,
            LSTM, LSTMNormalied, LinearNeuron)}
_ROOT = Path(__file__).resolve().parents[1]


def code_version():
    digest = hashlib.sha256()
    files = [_ROOT / 'run_animal_learning.py']
    for directory in ('nn', 'networks', 'envs', 'experiment'):
        files.extend(sorted((_ROOT / directory).glob('*.py')))
    for path in files:
        digest.update(path.relative_to(_ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def _atomic(path, write):
    path = Path(path)
    descriptor, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_state(path, state, metadata):
    arrays, objects, seen = {}, {}, {}

    def encode(value, name):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object arrays are not supported')
            arrays[name] = value.copy()
            return {'array': name}
        if isinstance(value, np.generic):
            return encode(value.item(), name)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {'dict': {str(k): encode(v, name + '/' + str(k)) for k, v in value.items()}}
        if isinstance(value, (list, tuple, deque)):
            return {type(value).__name__: [encode(v, name + '/' + str(i)) for i, v in enumerate(value)]}
        key = str(id(value))
        if key in seen:
            return {'ref': seen[key]}
        number = str(len(objects))
        seen[key] = number
        objects[number] = {}
        if isinstance(value, np.random.Generator):
            objects[number] = {'class': 'Generator', 'state': encode(value.bit_generator.state, name + '/rng')}
        elif type(value).__name__ in _CLASSES and type(value) is _CLASSES[type(value).__name__]:
            objects[number] = {'class': type(value).__name__,
                               'state': encode(vars(value), name)}
        else:
            raise ValueError(f'Unsupported checkpoint object: {type(value).__name__}')
        return {'ref': number}

    document = {'schema': 1, 'metadata': metadata, 'state': encode(state, 'state'), 'objects': objects}
    arrays['__metadata__'] = np.array(json.dumps(document, allow_nan=False))
    _atomic(path, lambda f: np.savez_compressed(f, **arrays))


def load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(str(archive['__metadata__']))
        if document['schema'] != 1:
            raise ValueError('Unsupported checkpoint schema')
        objects = {}
        for key, record in document['objects'].items():
            cls = record['class']
            if cls == 'Generator':
                objects[key] = np.random.default_rng()
            elif cls in _CLASSES:
                objects[key] = _CLASSES[cls].__new__(_CLASSES[cls])
            else:
                raise ValueError(f'Unknown checkpoint class: {cls}')

        def decode(value):
            if not isinstance(value, dict):
                return value
            if 'array' in value:
                return archive[value['array']].copy()
            if 'ref' in value:
                return objects[value['ref']]
            if 'dict' in value:
                return {k: decode(v) for k, v in value['dict'].items()}
            for kind, cls in (('list', list), ('tuple', tuple), ('deque', deque)):
                if kind in value:
                    return cls(decode(v) for v in value[kind])
            raise ValueError('Invalid checkpoint value')

        for key, record in document['objects'].items():
            obj = objects[key]
            values = decode(record['state'])
            if record['class'] == 'Generator':
                obj.bit_generator.state = values
            else:
                obj.__dict__.update(values)
        return decode(document['state']), document['metadata']


@contextmanager
def writer_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.writer.lock').open('a+b') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another process is writing this experiment') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Checkpoints:
    def __init__(self, experiment, metrics):
        self.experiment = experiment
        self.metrics = metrics
        self.directory = Path('results') / experiment.database_name
        self.record = self.directory / f'run_{experiment.run}.json'
        self.metadata = {'params': experiment.args_for_run, 'code_sha256': code_version(),
                         'python': platform.python_version(), 'numpy': np.__version__}
        self.history = []

    def status(self, status, next_step):
        document = {**self.metadata, 'status': status, 'next_step': next_step,
                    'updated_utc': datetime.now(timezone.utc).isoformat(),
                    'checkpoints': self.history}
        _atomic(self.record, lambda f: f.write(json.dumps(document, indent=2).encode('utf-8')))

    def save(self, state, status='running'):
        files = {}
        for metric in self.metrics:
            metric.commit_values()
            with open(metric.path, 'rb+') as f:
                data = f.read()
                f.flush()
                os.fsync(f.fileno())
            files[Path(metric.path).name] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        step = state['next_step']
        path = self.directory / f'checkpoint_{self.experiment.run}_{step}.npz'
        save_state(path, state, {**self.metadata, 'csv': files})
        self.history.append({'file': path.name, 'step': step,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.status(status, step)
        return path

    def restore(self, path):
        path = Path(path)
        if path.resolve().parent != self.directory.resolve():
            raise ValueError('Checkpoint must belong to the current experiment directory')
        if self.record.exists():
            self.history = json.loads(self.record.read_text(encoding='utf-8')).get('checkpoints', [])
            entry = next((r for r in self.history if r['file'] == path.name), None)
            if entry and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Checkpoint checksum does not match the run record')
        try:
            state, metadata = load_state(path)
        except (KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ValueError('Invalid or damaged checkpoint') from exc
        if any(metadata.get(k) != v for k, v in self.metadata.items()):
            raise ValueError('Checkpoint parameters, code or runtime do not match')
        if not 0 <= state['next_step'] <= self.experiment.get_int_param('steps'):
            raise ValueError('Invalid checkpoint progress')
        # Validate all files before removing records produced after the snapshot.
        repairs = []
        for metric in self.metrics:
            data = Path(metric.path).read_bytes()
            info = metadata['csv'][Path(metric.path).name]
            prefix = data[:info['size']]
            if len(prefix) != info['size'] or hashlib.sha256(prefix).hexdigest() != info['sha256']:
                raise ValueError('CSV data before checkpoint was changed or lost')
            tail = data[info['size']:]
            for row in csv.reader(io.StringIO(tail.decode('utf-8'))):
                if len(row) != len(metric.db_columns) or row[0] != str(self.experiment.run):
                    raise ValueError('Cannot roll back CSV records belonging to another run')
            if tail:
                repairs.append((metric.path, prefix))
        for target, data in repairs:
            _atomic(target, lambda f, data=data: f.write(data))
        self.history = [r for r in self.history if r['step'] <= state['next_step']]
        self.status('running', state['next_step'])
        return state
