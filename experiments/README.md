# Online sequence-model comparison

This directory compares value predictors on the existing trace-patterning
environment. It adds RNN, GRU, LSTM, a causal Transformer and a GTrXL adaptation
alongside the existing Columnar/CCN implementation. It does not reproduce
language-model or DMLab scores from the Transformer papers.

## Sources and implementation scope

| Source | What is used here |
| --- | --- |
| [Rafiee et al., From Eye-blinks to State Construction](https://arxiv.org/abs/2011.04590) | Online prediction, trace patterning and recurrent baselines; current repository environment reused |
| [Javed et al., CCN, JMLR 2023](https://jmlr.org/papers/volume24/23-0367/23-0367.pdf) | Existing columnar/constructive LSTM and RTRL/TD traces |
| [Dai et al., Transformer-XL, ACL 2019](https://aclanthology.org/P19-1285/) | Detached layer memories and relative attention in `gtrxl.py` |
| [Parisotto et al., GTrXL, ICML 2020](https://proceedings.mlr.press/v119/parisotto20a.html) | Pre-normalization, ReLU branches, GRU gates, gate bias initialized to 2 |

The ordinary Transformer uses sinusoidal positions and causal pre-normalized
attention over a rolling raw-observation window. GTrXL implements content and
relative-position attention scores explicitly, rather than the relative-shift
optimization. Its per-query attention horizon is limited to `memory` previous
positions, so segment replay and one-step inference agree at fixed parameters.
This sliding mask is an adaptation to one-step updates; it differs from the
original fixed-segment Transformer-XL schedule. Dropout is disabled in all
baselines. GTrXL has a scalar value head, not V-MPO, an action policy or DMLab.

## Install and run

Run commands from the repository root:

```powershell
uv sync --extra experiments
uv run --extra experiments python experiments/verify.py --cuda
uv run --extra experiments python experiments/run.py --preset all --name pilot --steps 20000 --device cpu
uv run --extra experiments python experiments/compare.py --runs "pilot_*" --output pilot_comparison
```

PyTorch 2.7.1 is pinned to the CUDA 12.6 index on Windows/Linux. This version is
chosen for the existing NVIDIA driver, not because it is the latest PyTorch.
The base NumPy project does not require this extra. CPU is the default: small
one-transition updates can run faster on CPU than GPU. CUDA must be measured,
not assumed to accelerate them. `--device auto` selects CUDA when available;
CCN remains NumPy/CPU even when a suite requests CUDA.

Examples for parameter sensitivity (each name must be new):

```powershell
uv run --extra experiments python experiments/run.py --preset lstm --name lstm_short --context 8 --steps 20000
uv run --extra experiments python experiments/run.py --preset lstm --name lstm_long --context 48 --steps 20000
uv run --extra experiments python experiments/run.py --preset lstm --name lstm_lr_high --lr 0.03 --steps 20000
uv run --extra experiments python experiments/run.py --preset transformer --name transformer_gpu --device cuda --steps 20000
```

`--hidden`, `--layers`, `--heads`, `--seed`, `--lr`, `--context`, `--memory`,
`--isi LOW HIGH`, `--iti LOW HIGH`, `--window`, `--clip`, `--steps-per-stage`,
`--features-per-stage`, `--normalization-rate`, `--log-every` and
`--checkpoint-every` are configurable. LSTM/GRU/RNN context controls the gradient
window, not how long recurrent state persists. Transformer context controls
available raw history. GTrXL context controls gradient replay, while memory
controls detached layer memory. These are not interchangeable memory budgets.

Presets: `rnn`, `gru`, `lstm`, `transformer`, `gtrxl`, `columnar`, `ccn`,
`ccn_current`, `ccn_paper`. The last two expose the old run's configuration and
a 10M-step paper-style CCN schedule; neither is a claim of reproduced scores.
The short CCN preset deliberately uses 5,000-step stages to exercise freezing;
this is much shorter than the paper and may freeze features before convergence.

## What is held equal

- Same environment seed, valid patterns, observations, random delays and noise.
  A chained SHA-256 of every observation, reward and evaluation target audits
  matching streams. Environment sampling does not use the model's RNG.
- Same gamma=.90 and lambda=.99; predict before the current update; use
  `delta = reward + gamma * next_prediction - previous_prediction`.
- One newly observed transition and one parameter update per step. No offline
  dataset, repeated epochs, sampled replay losses or future observation access.
- The exact environment target is used only for scoring, never as a training
  label. A prior-only running target mean is an evaluation baseline, too.

The PyTorch models recompute the previous value's derivative on the recent
window under current parameters, starting at a detached saved boundary state.
They accumulate `e = gamma * lambda * e + grad(value)` and apply RMSProp to
`-delta * e`. RMSProp squared averages start at one, alpha=.9999, epsilon=1e-8.
Readout weights start at zero. LSTM's combined forget bias starts at one.
Persistent state is advanced once under the pre-update parameters; it is never
reset at a new cue. Historical boundary states therefore contain features from
older parameter versions, a standard limitation of online truncated replay.

CCN retains its original per-cell/readout update rules, readout learning-rate
scaling, feature normalization and RTRL traces. It clamps the active stage at
the last available stage. PyTorch models additionally clip the TD-gradient
norm at 10 (configurable); CCN does not clip. Consequently this is an
architecture-plus-learning-rule comparison, not a pure architecture ablation.
Parameter counts are recorded but are not matched. It is also not the paper's
4,000-operation budget benchmark. A second-stage study can match budgets and
investigate initialization, optimizer and clipping independently.

## Results and recovery

Every run writes to `results/<name>/`:

- `metrics.csv`: rolling MSE and zero/mean baselines; conditional errors and
  predictions; TD error; out-of-range fraction; clipping fraction; parameter,
  eligibility and memory norms; forget-gate diagnostics; speed.
- `predictions.csv`: every tenth step plus cues and rewards, including cue,
  age and waiting phase. Profiles from this sample are diagnostic, not the
  full-stream MSE.
- `checkpoint_<step>.pt`: all weights, optimizer state, eligibility traces,
  observation windows, boundary/persistent states, environment and RNG states,
  error window, previous prediction and committed CSV hashes.
- `run.json`: configuration, parameter count, versions, source hash, device,
  progress/status, timing, checkpoint checksums and completed-stream digest.

All intermediate checkpoints are retained. Torch payloads contain tensors and
primitive containers only and load with `weights_only=True`; CCN is embedded
as the existing safe NPZ representation. Writes are atomic and an OS writer lock
prevents concurrent writers. Resume validates config, versions, source,
checkpoint checksum and both metric prefixes before truncating later logs.

```powershell
uv run --extra experiments python experiments/run.py --preset lstm --name pilot_lstm --steps 20000 --resume results/pilot_lstm/checkpoint_000010000.pt
```

Repeat the original parameter overrides when resuming. Source changes require
a new run. CPU verification checks exact resumed predictions, parameters and
non-timing metrics; GPU runs need not be bit-identical. Timing excludes import
and model construction, includes most logging/checkpoint overhead, and omits
the last snapshot's own write time. GPU peak allocation is recorded, not total
system RAM. No hardware speed superiority should be inferred from mixed devices.

## How to read the diagnostics

An MSE/zero ratio below one is necessary evidence of improvement over zero, but
not proof of useful memory: a constant mean predictor may do better too. Check
valid/invalid waiting errors and cue-age profiles. A meaningful predictor must
distinguish valid cues after they disappear and anticipate delayed rewards.

Forget gates are sampled from the first LSTM layer and from all CCN cells.
Values near one suggest slower state decay, but do not demonstrate that state
contains useful information. Parameter/trace norms depend on model size.
CCN's recorded trace norm is readout-only, and its gradient norm is left blank
rather than compared incorrectly against whole-model PyTorch gradients.
Parameter change is the norm of the difference since the preceding log, not
the sum of individual update lengths. Gate/gradient diagnostics are sampled
every 100 steps; error metrics use all steps in the rolling window.

Use short pilots to detect instability and estimate runtime. Strong claims
require several seeds, equally thorough tuning, longer training and resource
matching. Short pilots are not evidence that a method cannot eventually learn.

## 第一阶段实测：2026-10-06

完成 13 组各 20,000 步的 CPU 探索实验，并完成 RTX 4060 上的 1,000 步
LSTM 运行及三类模型的 CUDA 更新验证。以下误差使用最后 10,000 步。
完整记录见 `results/phase1_20261006_comparison.csv`，曲线和线索后预测图
分别为同名前缀的 `.png` 与 `_profiles.png`。

| 模型（seed=0，默认学习率 0.001） | 参数量 | 末窗口 MSE |
| --- | ---: | ---: |
| CCN（4 个特征／阶段，5,000 步／阶段） | 1,297 | 0.018684 |
| 固定 Columnar | 913 | 0.018825 |
| RNN | 497 | 0.019326 |
| GRU | 1,457 | 0.019173 |
| LSTM | 1,937 | 0.019229 |
| 因果 Transformer | 3,537 | 0.020565 |
| GTrXL 迁移实现 | 6,833 | 0.019087 |

相同数据的零预测误差是 0.019969，历史均值基线是 0.018637。
所有默认模型都没有优于均值基线，有效／无效等待期的平均预测差仅约
0.0004–0.0015。预测曲线尚未显示随有效线索等待时间明显上升，不能
认为任何架构在这次短测中已充分学会任务。CCN 的小幅领先不构成最终
准确率或效率优势的证明，尤其本轮没有匹配计算预算，部分短测并行运行。

LSTM 学习率 0.001、0.003、0.03 的误差分别为 0.019229、0.020392、
0.033998。学习率 0.03 时末窗口约 38.1% 的预测小于 0 或大于 1；
这是超出评分目标范围的比例，不能当作分类错误率。该配置等待期预测
接近零，无奖励区间的 MSE 却从默认配置的 0.001227 升至 0.018280。
较大的更新在这组设置下没有带来更好的提前预测。

LSTM 窗口 8／48 步的误差仅相差约 2.2e-7。它只说明当前初始化、
TD 资格迹和短训练下没有显著差异，不代表长窗口没有价值。默认配置
末次遗忘门均值约 0.727；输入权重变化范数约 0.000407，而输出权重
变化约 0.027869。特征提取部分变化较小、预测接近常数是观察；
“记忆保持不足或早期信用分配较弱”仍是需要进一步实验验证的解释。

旧配置 `ccn_current` 的误差约 0.042978，但它同时改变特征数、学习率、
阶段长度和归一化，不能把差距归因于其中某一个参数。LSTM／CCN 又各做了
seed=1 的短测，均未优于该种子的均值基线；两种子不足以建立统计结论。

首轮探索后修补了“日志和诊断间隔不对齐时，恢复应保留上次诊断值”的
记录细节。它没有改变参数更新或上表数据。首轮快照保存着修补前的来源
指纹，因此严格恢复检查会拒绝用修补后的代码续训这些快照；它们仍可
用于权重分析。新运行和最终验证使用修补后的完整快照格式。
后续应先检验输出层／循环层的学习率、遗忘门初始化和更充分的训练，
再匹配资源预算；目前不宜宣布 LSTM、Transformer 或 CCN 获胜。
