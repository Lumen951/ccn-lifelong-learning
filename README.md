# Scalabale Real-Time Recurrent Learning Using Columnar-Constructive Networks (JMLR)

**Paper:** [link here](https://jmlr.org/papers/volume24/23-0367/23-0367.pdf)

**ALE policy evaluation dataset:** [link here](https://github.com/khurramjaved96/atari-prediction-benchmark)

## Python port

The network code and the animal-learning experiment are ported to Python (package
managed by [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                                              # create .venv and install numpy
uv run python run_animal_learning.py --config experiment_configs/animal_learning_camera_ready.json --run 0
uv run python count_configs.py --config experiment_configs/animal_learning_camera_ready.json
uv run python verify_gradients.py                    # finite-difference checks of the ported gradients
```

Results are written to CSV under `results/<experiment-name>/` (`runs.csv`,
`error_table.csv`, `predictions.csv`) instead of the MariaDB tables used by the
C++ version.

Runs under the same experiment name share one parameter header, including all
algorithm branches. Parameters not used by a branch are left blank. Run IDs
must be within the configuration's grid range; an existing run ID is rejected
before training to prevent duplicate records. To repeat an experiment, change
its `experiment.name`. Existing files with incompatible headers are rejected
rather than rewritten. Run experiments sequentially when sharing these CSV files.

Configuration files accept UTF-8 with or without a BOM; output CSV uses UTF-8.
Missing arguments, invalid dimensions and non-finite predictions produce a
nonzero exit code. On interruption or failure, buffered metrics are flushed;
resume from the most recent valid checkpoint as described below.
The original entry point's fixed ISI `(24, 36)`, ITI
`(80, 120)` and gamma `0.90` are retained; an `ISI` grid parameter does not
change the environment's interval in this port.

Notes:
- Config -> run parameter mapping reproduces the C++ version exactly (same grid
  expansion order), so `--run N` selects identical hyper-parameters in both.
- The port is pure Python + numpy and prioritizes faithfulness over speed; a
  50M-step run takes hours rather than minutes.
- Dead code in the C++ experiment path (Synapse graph machine, cartpole/tmaze/
  weather/copy_task/mountain_car envs, the embedded Atari emulator) is not ported.

### Algorithm validation and corrections in the Python port

`verify_gradients.py` checks nonzero readout weights and every dense LSTM
parameter against finite differences. Truncated windows use fixed initial
states, including warm-up, repeated buffer wraps and truncation length one.
It also checks RTRL, normalized RTRL with fixed statistics, RMSProp, gradient
decay, state reset and layer freezing. Running normalization statistics are
treated as constants by the gradient algorithm; their update is checked
separately.

The Python BPTT cache preserves each transition's preceding hidden and cell
state. The original C++ ring loses the predecessor of the oldest transition
after wrapping. Python also avoids backpropagating through unfilled slots and
resets the history counter when resetting state. These are deliberate fixes,
so gradients at those boundaries differ from the original implementation.
Input observations are copied into the cache to avoid mutation by an environment.

Python initializes cell states and RMSProp accumulators explicitly. The C++
LSTM constructor leaves hidden/cell states uninitialized, and the normalized
network constructor omits readout RMSProp initialization. Fixed initial states
and weights are required for a meaningful future comparison with C++.
The deterministic C++ probe checks eight RTRL cell steps and three dense LSTM
steps (before the original ring bug occurs), including state, gradients and
RMSProp parameter updates. Maximum absolute differences are approximately
`5.27e-8` and `2.10e-8`. This is a component comparison, not full experiment
equivalence; long experiment runs have not yet been validated.

### Checkpoints and experiment records

Every run saves an initial snapshot, a snapshot every 50,000 completed steps,
and a final snapshot. Set `--checkpoint-every N` to change the interval. All
snapshots are retained under the existing experiment result directory:

- `checkpoint_<run>_<step>.npz`: weights, optimizer accumulators, gradient
  traces, normalization statistics, network/environment state, RNG state,
  completed step count, previous prediction and the rolling error window.
- `run_<run>.json`: selected parameters, Python/NumPy versions, source SHA-256,
  status, progress, checkpoint filenames and checksums.
- `.writer.lock`: an OS lock preventing simultaneous writers to shared CSVs;
  the OS releases the lock on process exit, even if the lock file remains.

```bash
uv run python run_animal_learning.py --config experiment_configs/animal_learning_camera_ready.json --run 0 --checkpoint-every 50000
uv run python run_animal_learning.py --config experiment_configs/animal_learning_camera_ready.json --run 0 --resume results/animal_learning_camera_ready/checkpoint_0_50000.npz
```

Resume requires identical parameters, training source and Python/NumPy versions.
NPZ loading uses `allow_pickle=False`; numeric arrays have descriptive keys such
as `state/network/prediction_weights`. Checkpoint writes use a flushed temporary
file followed by atomic replacement. CSV prefix hashes bind snapshots to their
metric history. Resuming an older checkpoint removes only later rows belonging
to that run, then replays them; it refuses rollback if another run wrote later
rows, or if earlier data was modified. Interrupted steps are never saved as a
valid snapshot: restart from the last completed checkpoint. All intermediate
snapshots are retained, so disk usage increases during long runs.

The verification script tests exact disk recovery for four network types and
CLI recovery across layer changes and BPTT window wraps, without duplicate CSV
rows. To additionally run the original C++ comparison (requires a compiler):

```bash
g++ -std=c++20 -O2 verify_cpp.cpp src/nn/lstm.cpp src/nn/neuron.cpp src/nn/synapse.cpp src/nn/dynamic_elem.cpp src/nn/utils.cpp src/utils.cpp src/nn/networks/lstm_bptt.cpp -o build/verify_cpp.exe
uv run python verify_gradients.py --cpp build/verify_cpp.exe
```

The compile command assumes `build/` exists. On Linux, use `build/verify_cpp`
instead. The C++ probe does not require MariaDB, OpenCV or ROM data.

## Compilation instructions (C++ original)

Prerequisites:
- A compiler with C++11 support
- Mariadb/MySQL
- OpenCV

The code can be compiled using `compile.sh` in the base directory. 


