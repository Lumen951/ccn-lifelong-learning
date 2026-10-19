# 代码与数据路径

[返回实现入口](README.md)

核查日期：2026-10-19。源码基点：407f115ed49ee17c07eb00dcea6fd1bf617be977；当前文档改动尚未提交。本页基于静态阅读，不代表运行验证。

## 两条训练路径

| 路径 | 配置与创建 | 训练和记录 |
| --- | --- | --- |
| [原入口](../../run_animal_learning.py) | [ExperimentJSON](../../experiment/config.py) 展开 experiment_configs 参数网格；[factory](../../networks/factory.py) 创建网络 | TracePatterning → backward/forward/update → [Metric](../../experiment/metric.py) → [NPZ 检查点](../../experiment/checkpoint.py) |
| [对照入口](../../experiments/run.py) | [configs.json](../../experiments/configs.json) 的 defaults + preset + CLI 覆盖；CCNLearner 或 OnlineLearner | 环境逐步采样 → TD 更新 → 滚动/条件指标 → [PT 检查点](../../experiments/checkpoint.py) |

原入口结果相对当前工作目录写入 results；对照入口以脚本解析出的仓库根目录为基准。操作说明要求从仓库根目录运行。
原入口用 run 编号选择网格项；对照入口使用 name 指定单次运行目录，两者不能直接互换。

## 共享实现

- [envs/animal_learning.py](../../envs/animal_learning.py)：TraceConditioning 与 TracePatterning；当前两训练入口使用后者。
- [nn/lstm.py](../../nn/lstm.py)：递归单元与梯度迹。
- [networks/incremental.py](../../networks/incremental.py)：分层特征、读出归一化与更新范围。
- [networks/bptt.py](../../networks/bptt.py)：NumPy dense LSTM / TBPTT。
- [experiments/models.py](../../experiments/models.py)、[gtrxl.py](../../experiments/gtrxl.py)：独立的 PyTorch 对照路径。
- [experiments/compare.py](../../experiments/compare.py)：已有运行的比较入口，说明见实验 README。

## 产物

原入口写 runs.csv、error_table.csv、predictions.csv、run_<run>.json 和 NPZ 快照。
对照入口写 metrics.csv、predictions.csv、run.json 和 PT 快照；CCN 状态以现有 NPZ 编码嵌入 PT。
results 被 Git 忽略，克隆源码不会获得本地结果。正式产物登记将在文档阶段 C 完成。

## 参考路径

[C++ 动物学习入口](../../experiment_animal_learning.cpp)、[src](../../src/) 和 [include](../../include/) 保留作参考。
不为整理文档移动代码。入口差异见 [实现状态](implementation-status.md)，选择理由见 [决策记录](decisions.md)。
