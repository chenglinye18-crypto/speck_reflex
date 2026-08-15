# SpikeMS pretrained reference reproduction — 2026-08-15

## Research Question

验证 SpikeMS 官方代码、官方预处理 EV-IMO 数据和官方 pretrained checkpoint，能否在
当前机器上输出 independent moving-object events。

本次结论：官方预处理数据无法从作者提供的唯一入口取得，Data Gate 失败。按照 gate 规则，
没有进入模型加载或推理，也没有用 EVIMO2 或自行转换的数据替代。

## Baseline Commit

`c55ffcf5fdf112ef0037d3f4c36ab1cf2620e9ec`

该 commit 已在 `main` 和 `origin/main`，开始复现时工作树 clean，且
`git diff --cached --check` 通过。

## SpikeMS Upstream

- SHA：`c449c83313423d62a23d92df32dd8d3180680a36`
- 状态：clean，未修改
- 官方 checkpoint：
  `third_party/SpikeMS/pretrainedModels/EVIMO-pretrained/out/checkpoint.pth.tar`
- checkpoint SHA-256：
  `d9cbff9c3d97a4a1dc95ce6bb1c08dbabb5b6338fcbd5c11a1a8e9067a65f4c2`

## Official Data

- 作者 README 中的唯一入口：
  `https://drive.google.com/drive/folders/1yrHUqYf0rWrfxbQILzKB9_kDYWF6yekd`
- 检查日期：2026-08-15
- HTTP 结果：`404 Not Found`
- `gdown 5.2.0` folder inventory：失败，无法取得目录内容
- GitHub releases：无 release 或数据资产
- README Git history：只有同一个 Google Drive URL，没有历史替代入口
- 公开 issue：`https://github.com/prgumd/SpikeMS/issues/8` 从 2025-04-11 起报告
  同一链接失效，检查时仍 open、无回复
- 计划 sequence：README 示例 `eval_wall/seq_00`
- 计划本地路径：`/home/speck/datasets/evimo_spikems_reference/`
- 实际下载大小：0 bytes
- HDF5 fields：无法检查
- mask format：无法检查

没有访问或修改 `/home/speck/datasets/evimo2`。

## Gate 1 — Data

`SPIKEMS_DATA_GATE=FAIL`

- event count：不可用
- polarity：不可用
- tensor shape：不可用
- physical window：不可用
- time mapping：`TIME_MAPPING_UNRESOLVED`

Exact blocker：作者提供的官方预处理 EV-IMO folder 已不存在或不再公开，无法获得完成
faithful reference inference 所必需的 HDF5 和 `depth_mask_*.png`。任务禁止使用 EVIMO2、
自行转换数据或非官方数据替代，因此必须在 Gate 1 停止。

## Gate 2 — Model

`NOT_RUN_DATA_GATE_FAILED`

静态审计确认：

- 预期 architecture：`2 -> 16 -> 32 -> 64 -> 32 -> 16 -> 2`
- checkpoint 文件存在且能被当前 PyTorch 以 CPU map-location 解析
- checkpoint keys：`epoch`、`loss`、`optimizer_state_dict`、`state_dict`
- checkpoint epoch：100
- `state_dict`：18 entries

以上不是 Model Gate PASS。没有在官方 sample 上加载网络或执行 forward，因此没有检查
missing/unexpected weights、参数有限性和 output shape。

## Gate 3 — Single Sample

`NOT_RUN_DATA_GATE_FAILED`

- sample：不可用
- raw events：不可用
- GT foreground：不可用
- prediction spikes：不可用
- IoU：不可用
- visualization paths：未生成

## Gate 4 — Small Run

`NOT_RUN_DATA_GATE_FAILED`

- requested samples：0
- valid samples：0
- filtered samples：0
- mean / median IoU：不可用
- ratio statistics：不可用

## GT-assisted Conditions

若官方数据恢复，faithful reference reproduction 将明确标记：

`REFERENCE_ONLY_GT_ASSISTED`

- crop：作者 loader 使用 GT foreground event density 确定裁剪中心
- filtering：作者 loader 使用 GT background/foreground ratio 过滤 sample
- implication：结果只代表作者原始、GT-assisted 测试路径，不能描述为 full-frame 或
  deployment-ready performance

本次没有运行这些操作。

## Environment

- OS：Ubuntu 22.04.5 LTS
- Python：3.10.12
- torch：2.10.0+cu128
- torchvision：0.25.0+cu128
- CUDA build：12.8
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB
- driver：577.02
- numpy：1.26.4
- h5py：3.16.0
- missing legacy dependencies：`slayerCuda`、`cv2`、`strictyaml`、`tensorboardX`
- compatibility patches：无

Legacy dependency blockers 尚未进入处理阶段，因为 Data Gate 先失败。没有强装 CUDA 10 或
Python 3.7，也没有修改 upstream。

## SPIKEMS_REFERENCE_REPRODUCTION

`FAIL`

失败原因仅是官方预处理数据不可获得；这次结果不能用于判断模型本身能否运行或预测质量。

## Next Recommended Step

向 SpikeMS 作者请求恢复同一官方预处理 `eval_wall/seq_00` 文件夹或提供带校验值的官方镜像。
取得 HDF5 与完整配套 mask 后，从 Gate 1 重新开始。不要用 EVIMO2 顶替本次 reference
reproduction。
