# 实现状态与证据边界

[返回实现入口](README.md) · [代码地图](code-map.md)

日期：2026-10-19。证据分为“本轮静态阅读”“已有文档报告”“待验证”。本轮没有重新执行算法或恢复测试，也没有重新计算历史指标。

## 本轮静态阅读

| 项目 | 观察到的实现 | 研究影响 |
| --- | --- | --- |
| 算法映射 | factory 的 hybrid 使用 IncrementalNetworks；hybrid_normalized 使用 NormalizedIncrementalNetworks；对照 CCNLearner 使用前者 | 不能仅凭 ccn 名称认定采用单元内部归一化版本 |
| stage 切换 | 原入口在 i % S == S-1 时先增 layer；对照入口取 min((i+1)//S, 最后一层) | 两者均在编号为 S 的那次更新前切换，需明确步数约定；容量用尽后的行为不同 |
| 最后一层 | 原入口继续增加 layer；对照入口停留在最后一层 | 前者可超出循环参数范围，读出仍按 hi 缩放步长；后者继续训练末层 |
| 归一化 | IncrementalNetworks.forward 更新所有列的 feature_mean/feature_std | 循环参数冻结不等于读出统计或功能固定；普通 forward 不等于无副作用评估 |
| 列状态 | 所有列参与 forward，更新权重范围由 layer 决定 | 尚未训练的列存在状态，不应把状态存在等同于可训练或有读出贡献 |
| 任务随机性 | 环境 RNG 同时产生 10 个有效模式与后续延迟、刺激、噪声 | 换环境 seed 会改变规则和序列；独立评估须单独定义 |
| 延迟配置 | 原入口固定 ISI=(24,36)、ITI=(80,120)、gamma=.90；对照入口读取解析后的配置 | 配置字段存在不保证原入口使用它 |
| 评分与更新 | 环境 get_target 用有效模式及 remaining_until_US 计算评分目标；TD 目标使用 reward + gamma * prediction | 本地评分目标与完整长期回报的关系仍需对照论文核查 |
| 指标 | 对照入口记录先前目标均值基线、有效/无效等待期指标；预测文件按步采样并补记线索/奖励 | 采样预测文件不能直接替代完整流 MSE；均值基线仅是评分参照 |
| 续训 | 对照入口检查配置、源码摘要、依赖版本、快照校验和 CSV 前缀 | 更改源码或总步数后不能假定可直接恢复旧快照 |

对应源码：[原入口](../../run_animal_learning.py)、[对照入口](../../experiments/run.py)、[网络](../../networks/incremental.py)、[环境](../../envs/animal_learning.py)、[factory](../../networks/factory.py)、[恢复](../../experiments/checkpoint.py)。

## 已有文档报告，未经本轮重验

[根 README](../../README.md) 记录 Python 的 BPTT 边界修正、显式初始化及局部 C++ 数值对照。局部对照不证明完整实验等价。
[实验 README](../../experiments/README.md) 的 2026-10-06 报告称默认短测模型未优于均值基线；这是历史报告，不能当作本轮重新分析结果。
CCN 为 NumPy/CPU 路径；PyTorch 对照可选择 CUDA。根 README 的 Python CPU 描述应结合其移植章节范围理解，不应扩大到全部对照模型。

## 尚未建立或未完成核查

目前阅读的训练入口没有独立列级消融、A/B 上下文任务序列或功能驱动冻结流程。
Python Atari 路径据现有 README 尚未移植；不能因为保留 C++ ALE 源码就声称已支持 Python Atari。
完整论文版本核对、评估协议、计算预算匹配及正式复现实验均待完成。
pyproject.toml 仍使用旧项目名；本轮未修改元数据。
