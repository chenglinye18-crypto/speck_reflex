# Research Principles

## 唯一核心问题

在 camera ego-motion 条件下，仅使用 DVS events，验证 SNN 是否能够区分：

- camera ego-motion 导致的背景事件；
- independent moving object 导致的事件。

SpikeMS 先作为科学可行性 reference baseline，不默认视为最终避障网络。

## 固定研究顺序

```text
Raw DVS
-> Event Representation
-> Motion Segmentation SNN
-> Ego-motion Background / Independent Motion
-> Motion Evidence
-> Risk / Approaching / Looming
-> Reflex Decision
```

当前只推进到 `DVS -> SNN -> Independent Motion`。上一层没有验证时，不提前实现下一层。

## Baseline 顺序

1. 明确 SpikeMS 的输入、event representation、网络、输出、loss 与评估方式。
2. 原样复现固定版本的 SpikeMS。
3. 在 EVIMO2 上建立可重复、可诊断的 baseline。
4. 分开进行三种基础实验：
   - Camera Static / Object Moving
   - Camera Moving / Object Static
   - Camera Moving / Object Moving
5. baseline 成立以后，才允许一次只改变一个关键因素。

禁止同时更换数据、event representation、网络和 loss。否则失败无法归因。

## 每一层必须可回答的问题

- 输入是什么？
- 具体做了什么？
- 输出是什么？
- 这一层解决什么现实问题？
- 用什么可观察结果判断它工作？

重要中间结果必须能够独立保存和检查，不能只提供 `events -> model -> result` 黑盒。

## 优先级

```text
科学正确性
> 可解释 / 可诊断
> 可复现
> 实验速度
> 性能优化
> 软件工程美观
```

本阶段不把 TTC、方向、急停、STM32、FPGA 或 Speck 部署混入核心算法验证。
