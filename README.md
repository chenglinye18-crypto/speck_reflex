# speck_reflex

本仓库当前只研究一个问题：

> 在 Event Camera 自身运动时，仅使用 DVS events，SNN 能否把 camera ego-motion
> 产生的背景事件与 independent moving object 产生的事件分离？

第一参考基线固定为
[SpikeMS: Deep Spiking Neural Network for Motion Segmentation](https://github.com/prgumd/SpikeMS)。
本阶段不设计新 SNN，不优化网络，也不讨论 TTC、risk、reflex decision 或硬件部署。

## 当前研究边界

```text
Raw DVS
  -> Event Representation
  -> SpikeMS motion-segmentation SNN
  -> ego-motion background / independent motion
```

下一阶段问题只有在这一层得到可信验证后才进入仓库。

## 仓库结构

```text
AGENTS.md                  对研究与开发工作的约束
RESEARCH_PRINCIPLES.md     当前科学问题、实验顺序和退出条件
third_party/SpikeMS/       固定版本的 GPL-3.0 upstream submodule；禁止修改
reference/spikems/         我们自己的复现记录、审计和后续适配说明
scripts/                   不修改 upstream 的辅助检查脚本
```

历史上的 Speck / Sinabs / Samna、N-MNIST、STM32N6、FPGA、hardware
backend、MCU/reflex protocol、旧 ANN/hybrid/自建 SNN、ego-motion head、synthetic
motion 以及 TTC/risk/collision pipeline 已退出 main 工作树。需要追溯时使用 Git history，
不要把它们复制回 `legacy/` 或 `archive/`。

## 获取固定的 SpikeMS reference

```bash
git clone --recurse-submodules <this-repository-url>
cd speck_reflex
git submodule update --init --recursive
bash scripts/verify_spikems_reference.sh
```

当前固定的 SpikeMS commit：
`c449c83313423d62a23d92df32dd8d3180680a36`。

运行 upstream 前请先阅读
[`reference/spikems/README.md`](reference/spikems/README.md)。原项目依赖较老的 Python、
PyTorch 和 CUDA 软件栈；本仓库当前没有宣称它已经在现代环境中复现成功。

## 研究状态

- 已完成：仓库主线清理；SpikeMS upstream 固定与静态审计。
- 尚未完成：原始预训练推理复现、EVIMO2 数据协议确认、三种基础实验。
- 当前禁止：在 baseline 成立前同时更换数据、event representation、网络或 loss。

清理前的仓库状态由 tag `pre-spikems-reset-20260815` 保存；没有改写 Git history。
