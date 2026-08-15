# SpikeMS reference baseline

## 角色与边界

`third_party/SpikeMS` 是 upstream GPL-3.0 项目的原样、固定版本：

- upstream：<https://github.com/prgumd/SpikeMS>
- pinned commit：`c449c83313423d62a23d92df32dd8d3180680a36`
- 许可证：`third_party/SpikeMS/LICENSE`

禁止直接修改 submodule。复现笔记、数据适配、诊断或 wrapper 只能写在
`reference/spikems/` 或顶层 `scripts/`。

## 它想解决什么现实问题？

Event Camera 在相机或物体运动时会产生大量事件。SpikeMS 尝试直接用时空脉冲表示，
输出属于独立运动物体的事件区域，从而将运动物体从背景事件中分离。

这与本项目的问题相关，但并不自动证明它在任意 camera ego-motion 下有效。必须用明确拆分的
静态/运动相机与静态/运动物体实验验证。

## 输入是什么？

当前 EVIMO loader 读取 HDF5：

```text
events       N x 4: [t, x, y, polarity]
events_idx   每个 mask frame 对应的事件起始索引
timeframes   [window_start, window_end, mask_image_index]
num frames   frame 数量
```

事件被转换成形状为 `[2, height, width, time_bins]` 的二值 spike tensor；两个通道表示
polarity，时间被归一化到离散 time bins。默认配置为 100 个 time bins，EVIMO 原始尺寸为
`260 x 346`，可选裁剪尺寸为 `128 x 128`。

## 网络具体做了什么？

`unetRNN6Layer_noBlock.SNN` 是一个六层、逐时间运行的 encoder-decoder SNN：

```text
2 polarity channels
-> Conv 16, stride 2
-> Conv 32, stride 2
-> Conv 64, stride 2
-> Deconv 32, stride 2
-> Deconv 16, stride 2
-> Deconv 2, stride 2
```

每层通过项目内置的 modified SLAYER 实现 PSP 和 spike dynamics。这里的两个输出通道
不是“背景/物体”softmax 类别；它们保留事件 polarity，并用输出脉冲表示被预测为运动物体的事件。

## 输出和判断方式是什么？

模型输出形状为 `[batch, 2, height, width, time_bins]` 的 spike tensor。upstream
`runner.py` 将 polarity、batch 和时间维求和得到二维预测区域，再与运动物体 mask 内的事件
求 IoU。保存图片时还会分别保存 input、ideal 和 prediction。

至少应保留以下可诊断中间结果：

- 输入 polarity/time-bin spike tensor；
- 原始标注 mask 和 mask 内事件；
- 每层 spike activity；
- 输出 spike tensor 与二维聚合结果；
- 每个样本的 background/object event ratio 与 IoU。

## Loss 的当前事实

upstream README 称预训练 EVIMO 模型使用了 Cross Entropy Loss 和 SpikeLoss；仓库内的
modified SLAYER `loss.py` 提供 `spikeTime`、`MembraneSpikeTime`、`numSpikes`、`MSE`
和 `getIOU`。但固定 commit 没有清晰的训练入口，`test.py` 只加载 checkpoint 并计算 IoU。

因此目前只能确认测试路径，不能仅凭仓库内容声称训练过程已可复现。下一步必须从论文、配置与
可获得的训练材料中逐项核对目标构造、组合 loss、优化器、数据拆分和 checkpoint 来源。

## 已识别的科学风险

EVIMO loader 在测试数据路径中使用 ground-truth mask 做了两项处理：

1. 根据 mask 内外事件比例执行 `maxBackgroundRatio` 过滤；
2. 启用 `--crop` 时，以 mask 内事件最密集位置决定裁剪中心。

这会筛选较容易样本，并让输入裁剪依赖测试真值。复现 upstream 报告时应原样保留并清楚标记；
评估真实 ego-motion 分离能力时，必须另外报告不使用这些真值辅助的结果。不能悄悄改 upstream，
应通过本仓库外置的 adapter/wrapper 做对照实验。

其他已知限制：

- upstream 环境记录为 Python 3.7、Ubuntu 18.04、CUDA 10、PyTorch 1.4；
- requirements 固定了较旧的软件版本并包含 CUDA 扩展；
- upstream README 的部分命令与目录名存在不一致，应以固定 commit 的实际文件为准；
- 当前仓库重置不等于 baseline 已经复现成功。

## 最小复现顺序

1. 验证 submodule pin 与 clean 状态：

   ```bash
   bash scripts/verify_spikems_reference.sh
   ```

2. 在隔离环境中按 upstream README 准备兼容依赖，不要把兼容性补丁写进 submodule。
3. 获取 upstream 指定的预格式化 EVIMO 数据，并记录校验值与来源。
4. 先用 upstream `test.py`、原始 checkpoint 和原始参数复现其测试路径。
5. 保存逐样本输入、标注、输出、过滤原因和 IoU，确认结果可重复。
6. 再建立 Camera Static/Object Moving、Camera Moving/Object Static、Camera
   Moving/Object Moving 三个互斥实验子集。
7. 最后增加无真值 crop/filter 对照；一次只改变这一项。

upstream README 中的测试命令和参数仍以 `third_party/SpikeMS/README.md` 为准。
