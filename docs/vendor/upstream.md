# 上游来源与使用边界

[返回来源入口](README.md)

日期：2026-10-19。本轮核查本地许可证、README 和参考路径；未联网核验远端 fork 关系、上游提交或论文全文。

| 来源 | 用途与证据 | 限制 |
| --- | --- | --- |
| [CCN JMLR 论文](https://jmlr.org/papers/volume24/23-0367/23-0367.pdf) | 根 README 提供的论文链接 | 具体版本、实验配置和目标定义需逐项核对 |
| [Atari prediction benchmark](https://github.com/khurramjaved96/atari-prediction-benchmark) | 根 README 提供的数据/策略评估参考 | 本轮未下载，不代表 Python Atari 已可运行 |
| 本地 [LICENSE](../../LICENSE) | MIT，Copyright (c) 2023 Khurram Javed | 保留原许可与署名，不把原实现标为新增研究原创 |
| [src](../../src/)、[include](../../include/)、[原 C++ 入口](../../experiment_animal_learning.cpp) | 已保留的参考实现 | 尚未锁定对应上游提交，不能宣称未修改的特定发布版 |
| [当前项目](https://github.com/Lumen951/ccn-lifelong-learning) | 本地 origin 指向的已重命名仓库 | origin 地址本身不证明 GitHub fork 关系 |

PyTorch 对照涉及的 Trace patterning、Transformer-XL 和 GTrXL 资料及适配限制，集中保留在 [实验 README](../../experiments/README.md)。
根 pyproject 描述带有 JMLR 2024，而实验 README 标注 JMLR 2023；书目信息存在不一致，后续对照论文正式元数据确认，不在本轮推断修正。

## 工作流参考

用户提供的《天猫技术大型 AI 项目的研发协同实践》启发了入口地图、规格与探索分离、决策留痕和验证检查。
原文链接：[大淘宝技术文章](https://mp.weixin.qq.com/s/0wnFz3G6niLetSQptUi-SA)。这是工作流参考，不是 CCN 算法证据；其示例不构成执行授权。本轮未重新联网核验文章。
