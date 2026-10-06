"""Atomic experiment snapshots, restricted loading and metric-prefix checks."""
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiment.checkpoint import _atomic, load_state, save_state

ROOT = Path(__file__).resolve().parents[1]
LOGS = ('metrics.csv', 'predictions.csv')


def source_digest():
    digest = hashlib.sha256()
    paths = [ROOT / 'experiments' / name for name in
             ('run.py', 'models.py', 'gtrxl.py', 'checkpoint.py')]
    for folder in ('envs', 'nn', 'networks', 'experiment'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode('utf-8'))
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def write_json(path, document):
    encoded = json.dumps(document, indent=2, allow_nan=False).encode('utf-8')
    _atomic(path, lambda stream: stream.write(encoded))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def pack_ccn(network, directory):
    descriptor, name = tempfile.mkstemp(dir=directory, suffix='.npz')
    os.close(descriptor)
    try:
        save_state(name, {'network': network}, {})
        return torch.from_numpy(np.frombuffer(Path(name).read_bytes(), dtype=np.uint8).copy())
    finally:
        Path(name).unlink(missing_ok=True)


def unpack_ccn(payload, directory):
    descriptor, name = tempfile.mkstemp(dir=directory, suffix='.npz')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload.cpu().numpy().tobytes())
        return load_state(name)[0]['network']
    finally:
        Path(name).unlink(missing_ok=True)


def prefix_records(directory):
    records = {}
    for name in LOGS:
        content = (directory / name).read_bytes()
        records[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    return records


def save_checkpoint(directory, state, config, manifest, status):
    payload = cpu_copy(state)
    payload.update(schema=1, config=config, source_sha256=source_digest(),
                   torch_version=str(torch.__version__), numpy_version=np.__version__,
                   python_version=platform.python_version(),
                   prefixes=prefix_records(directory), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    name = f"checkpoint_{state['next_step']:09d}.pt"
    path = directory / name
    _atomic(path, lambda stream: torch.save(payload, stream))
    record = {'file': name, 'step': state['next_step'],
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    records = [r for r in manifest.get('checkpoints', []) if r['file'] != name]
    records.append(record)
    manifest.update(status=status, next_step=state['next_step'],
                    elapsed_seconds=state['elapsed'], checkpoints=records)
    write_json(directory / 'run.json', manifest)
    return path


def restore_checkpoint(path, directory, config, device):
    path = Path(path).resolve()
    if path.parent != directory.resolve():
        raise ValueError('Resume checkpoint must belong to the selected output directory')
    manifest = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    record = next((r for r in manifest['checkpoints'] if r['file'] == path.name), None)
    if record is None or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Checkpoint checksum mismatch or checkpoint not registered')
    # Only tensors and primitive dictionaries/lists are accepted, never models/pickle globals.
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['schema'] != 1 or payload['config'] != config:
        raise ValueError('Checkpoint configuration differs from this run')
    if payload['source_sha256'] != source_digest():
        raise ValueError('Training source changed; use a new run name')
    if payload['torch_version'] != str(torch.__version__) or payload['numpy_version'] != np.__version__:
        raise ValueError('Dependency versions changed')
    if payload['python_version'] != platform.python_version():
        raise ValueError('Python version changed')
    # Validate both histories before rolling either back.
    for name, prefix in payload['prefixes'].items():
        content = (directory / name).read_bytes()
        if len(content) < prefix['bytes'] or hashlib.sha256(
                content[:prefix['bytes']]).hexdigest() != prefix['sha256']:
            raise ValueError(f'Metric prefix changed: {name}')
    for name, prefix in payload['prefixes'].items():
        with (directory / name).open('r+b') as stream:
            stream.truncate(prefix['bytes'])
            stream.flush()
            os.fsync(stream.fileno())
    torch.set_rng_state(payload['torch_rng'].cpu())
    if payload['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng']])
    return payload, manifest
