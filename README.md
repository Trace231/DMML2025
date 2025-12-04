# 图像分割基准套件

这是一个完整的图像分割评测项目，用于在一个小型测试集上比较多种类型的分割模型，并统一输出评估指标。本项目默认使用 **CrackForest** 道路裂缝数据集（共 118 张图像），支持自动下载、数据集拆分、模型训练/微调、推理、指标计算和结果归档。

## ✨ 功能亮点

- ✅ 支持的数据驱动基准：CRF 特征模型、CNN、Transformer、随机游走（扩散思想）、混合 CNN-Transformer、CNN-CRF、任意模型 + CRF 后处理。
- ✅ 自动化流水线：读取配置 → 下载/加载数据 → 构建模型 → 评测 → 导出 JSON/CSV 结果。
- ✅ 多指标评估：Pixel Accuracy、mIoU、Precision、Recall、F1、Dice 等。
- ✅ 结构化代码：模块化的 `src/segmentation_benchmark` 包，便于扩展自定义模型或数据集。
- ✅ 单元测试覆盖基础组件（指标计算、数据管线、注册表）。

## 📦 目录结构

```
segmentation-benchmark/
├── configs/                  # YAML 配置（默认 crackforest_benchmark.yaml）
├── data/                     # 数据集下载目录（首次运行自动生成）
├── scripts/                  # 命令行脚本（下载数据、运行基准等）
├── src/segmentation_benchmark/
│   ├── data/                 # 数据集加载与拆分
│   ├── evaluation/           # 评测器与注册表
│   ├── metrics/              # 指标计算
│   ├── models/               # 各类分割模型封装
│   └── utils/                # 配置与路径工具
├── tests/                    # Pytest 测试用例
├── reports/                  # 评测输出（自动创建）
├── artifacts/                # 训练权重等（占位目录）
├── requirements.txt          # 依赖列表
└── pyproject.toml             # 包配置
```

## 🗂️ 数据集说明

- **名称**：CrackForest Dataset（118 张城市道路裂缝图像）
- **官方地址**：<https://github.com/cuilimeng/CrackForest-dataset>
- **许可**：仅供非商业科研使用，请按项目 README 引用相关论文。
- **准备方式**：执行脚本 `python scripts/download_crackforest.py`，或在运行基准脚本时自动下载。

> 默认配置会将数据集划分为 Train:Val:Test = 60% : 20% : 20%。可以通过 YAML 配置自定义。

## 🔧 安装
注意本项目只能在python=3.10运行，pydensecrf请自行编译安装
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
#安装开发依赖
pip install -e .[dev]
```

> Windows PowerShell 下请使用如上命令；其他平台请自行调整虚拟环境激活方式。

## 🚀 快速上手

1. **下载数据集（可选）**：
   ```powershell
   python scripts/download_crackforest.py
   ```

2. **运行完整基准**：
   ```powershell
   python scripts/run_benchmark.py --config configs/crackforest_benchmark.yaml
   ```

   运行结束后，所有模型的指标会保存到 `reports/<run_name>/` 下：

   - `<model>_metrics.json`：单模型详细指标
   - `benchmark_summary.csv` / `benchmark_summary.json`：所有模型对比表

3. **自定义参数**：
   - `--device cuda` 指定在 GPU 上运行（若可用）。
   - `--skip-train` 跳过所有模型的训练/微调阶段，仅做推理。
   - `--save-predictions` 将每个模型的预测掩码保存为 `.npy` 文件。

## 🧠 模型族群概览

| 类型 | 注册名称 | 描述 |
| ---- | -------- | ---- |
| 特征 + CRF | `classical_crf` | 手工特征 + Random Forest + DenseCRF |
| CNN | `fcn_resnet50`, `deeplabv3_resnet50` | Torchvision 语义分割骨干，可微调 |
| Transformer | `segformer_b0` | HuggingFace SegFormer-B0 模型 |
| Diffusion 风格 | `random_walker` | 基于随机游走（扩散思想）的快速分割 |
| Hybrid | `hybrid_unet_transformer` | 自定义 CNN + 多头自注意力混合模型 |
| CNN-CRF | `cnn_crf` | CNN 预测 + DenseCRF 端到端组合 |
| 任意模型 + CRF 后处理 | `crf_wrapper` | 包装任意注册模型并追加 DenseCRF 后处理 |

> 所有模型均通过 `segmentation_benchmark.evaluation.registry` 注册，可轻松扩展。

## 📊 评估指标

默认输出以下指标：

- Pixel Accuracy
- Mean IoU
- Mean Precision / Recall / F1
- Mean Dice
- Per-class IoU / Precision / Recall / F1 / Dice
- Confusion Matrix

指标计算实现位于 `src/segmentation_benchmark/metrics/metrics.py`，可按需扩展。

## ⚙️ YAML 配置要点

`configs/crackforest_benchmark.yaml` 中的关键字段：

```yaml
dataset:
  root: data/crackforest  # 数据目录
  download: true          # 若缺失则自动下载
  image_size: 256         # 统一分辨率
  train_ratio: 0.6        # 训练集占比
  val_ratio: 0.2
  num_classes: 2

models:
  - name: fcn_resnet50
    builder: fcn_resnet50
    params:
      finetune_epochs: 1
      pretrained: true
  - name: fcn_resnet50_crf_post
    builder: crf_wrapper
    params:
      base_builder: fcn_resnet50
      crf_params:
        iterations: 5
```

如需新增模型，只需在 `models` 列表中追加一个条目，并确保对应的 `builder` 已注册。

## 🧩 扩展指南

- **添加新模型**：在 `src/segmentation_benchmark/models/` 中编写派生自 `BaseSegmenter` 的类，并使用 `@register_segmenter("your_name")` 装饰器注册。
- **替换数据集**：实现新的 Dataset 类和 `create_dataloaders` 工厂，并在配置中引用。
- **自定义指标**：在 `metrics` 模块中扩展 `SegmentationMetrics` 或 `MetricsAggregator`。

## ✅ 测试

```powershell
pytest -q
```

- `tests/test_metrics.py`：验证指标计算逻辑
- `tests/test_data.py`：验证数据集加载与 DataLoader 拆分
- `tests/test_registry.py`：确保关键模型已注册

## 📄 许可与引用

- 项目代码默认采用 MIT License（可在 `pyproject.toml` 中调整）。
- 使用 CrackForest 数据集时请遵循其非商业许可，并在论文/报告中引用原作者提供的文献。

## 🙏 致谢

感谢 CrackForest 数据集作者以及开源社区（PyTorch、Torchvision、Transformers、scikit-image、pydensecrf 等）提供的优秀工具。

祝你在道路裂缝分割任务中旗开得胜！🛣️✨
