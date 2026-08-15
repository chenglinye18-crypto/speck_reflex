# SNN Reflex 软件研究准则

## 当前唯一核心问题

当前阶段首先验证：

> **仅使用DVS事件，在Camera自身运动时，SNN能否区分Camera ego-motion产生的背景事件与Independent Moving Object产生的事件。**

当前主线参考 **SpikeMS**。

SpikeMS首先作为**科学可行性Baseline**，不默认视为最终避障网络。

---

## 研究顺序

严格按照：

```text
Raw DVS
→ Event Representation
→ Motion Segmentation SNN
→ Ego-motion Background / Independent Motion
→ Motion Evidence
→ Risk / Approaching / Looming
→ Reflex Decision
```

逐层推进。

**上一层没有验证，不提前设计下一层。**

当前重点只做到：

```text
DVS → SNN → Independent Motion
```

暂不把TTC、方向、急停、STM32、FPGA、Speck部署等问题混入核心算法验证。

---

## Baseline原则

优先：

1. 理解SpikeMS输入、网络、输出和Loss；
2. 复现SpikeMS；
3. 在EVIMO2上建立可信Baseline；
4. 完成Camera Static/Object Moving、Camera Moving/Object Static、Camera Moving/Object Moving三个基础实验；
5. Baseline成立以后再修改网络。

禁止同时：

```text
换数据
+ 换Event表示
+ 换网络
+ 换Loss
```

否则实验失败时无法定位原因。

---

## 软件设计原则

代码按照**科学问题**拆分，而不是按照未来硬件拆分。

每个模块必须明确：

```text
输入是什么？
输出是什么？
这一层解决什么问题？
怎么判断它是否工作？
```

重要中间结果必须能够单独观察和保存。

不要把整个算法写成：

```text
events → model → result
```

的黑盒。

---

## 实验原则

科研正确性优先级：

```text
科学正确性
> 可解释 / 可诊断
> 可复现
> 实验速度
> 性能优化
> 软件工程美观
```

每增加一个算法模块，都必须能够回答：

> **它解决了哪个明确问题？**

如果无法回答，就暂时不加入。

---

## 对话与开发原则

项目涉及新的算法概念时，优先先解释：

1. 它想解决什么现实问题；
2. 输入是什么；
3. 它具体做了什么；
4. 输出是什么；
5. 为什么这样做可能有效；

再讨论公式和代码。

不要默认使用者已有计算机视觉、深度学习或SNN算法背景。
