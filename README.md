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
Full C++ equivalence and long experiment runs have not yet been validated.

## Compilation instructions (C++ original)

Prerequisites:
- A compiler with C++11 support
- Mariadb/MySQL
- OpenCV

The code can be compiled using `compile.sh` in the base directory. 


