# Image Segmentation Benchmark Suite

This is a comprehensive image segmentation evaluation project for comparing various types of segmentation models on datasets and outputting unified evaluation metrics. We aim to validate the practical effectiveness of CRF post-processing techniques in image segmentation post-processing through this work. This project supports evaluation modules for two default datasets: the **"crackforest"** dataset and the **voc_2012** dataset, with support for automatic download, dataset splitting, model training/fine-tuning, inference, metric calculation, and result archiving.

## Key Features

- Supported data-driven baselines: CRF feature models, CNN, Transformer, DDP (Diffusion Model), Hybrid CNN-Transformer, CNN-CRF, and any model + CRF post-processing.
- Automated pipeline: Read configuration → Download/Load data → Build models → Evaluate → Export JSON/CSV results.
- Multi-metric evaluation: Pixel Accuracy, mIoU, Precision, Recall, F1, Dice, etc.
- Structured code: Modular `src/segmentation_benchmark` package for easy extension of custom models or datasets.
- Unit tests covering basic components (metric calculation, data pipeline, registry).

## File Tree

```
segmentation-benchmark/
├── configs/                  # YAML configurations (default: crackforest_benchmark.yaml)
├── data/                     # Dataset download directory (auto-generated on first run)
├── scripts/                  # Command-line scripts (download data, run benchmarks, etc.)
├── src/segmentation_benchmark/
│   ├── data/                 # Dataset loading and splitting
│   ├── evaluation/           # Evaluators and registry
│   ├── metrics/              # Metric calculation
│   ├── models/               # Various segmentation model wrappers
│   └── utils/                # Configuration and path utilities
├── tests/                    # Pytest test cases
├── reports/                  # Evaluation outputs (auto-created)
├── artifacts/                # Training weights, etc. (placeholder directory)
├── requirements.txt          # Dependency list
└── pyproject.toml             # Package configuration
```

## Dataset Information

### CrackForest Dataset

- **Dataset Name**: CrackForest Dataset (contains 118 urban road crack images)
- **Official Source**: <https://github.com/cuilimeng/CrackForest-dataset>
- **Usage License**: This dataset is limited to non-commercial research purposes. Please follow the citation requirements in the project README when using it.
- **Data Preparation**: Can be manually downloaded by running `python scripts/download_crackforest.py`, or automatically downloaded when running benchmark tests.

### Pascal VOC 2012 Dataset

- **Dataset Name**: Pascal VOC 2012 (contains semantic segmentation annotations for 21 classes)
- **Official Source**: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>
- **Usage License**: The Pascal VOC dataset follows its official usage license, typically allowing use for academic research.
- **Data Preparation**: When `download: true` is set in the configuration file, the system will automatically download the dataset via torchvision; the dataset will be saved to the `data/voc/VOCdevkit/VOC2012/` directory.
- **Note**: The dataset download website is often unstable. If you need the dataset, please contact: wangfeiming@mail.nankai.edu.cn, and we will seek ways to make the dataset publicly available!

> The default configuration splits the dataset into Train:Val:Test = 60% : 20% : 20%. This can be customized via YAML configuration.

## Installation

This project requires Python 3.10 environment. Please note that pydensecrf needs to be compiled and installed separately. **When configuring, note that Cython needs to be downgraded to version 2.X for compilation.**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# Install development dependencies
pip install -e .[dev]
```

> Use the above commands in Windows PowerShell; adjust the virtual environment activation method for other platforms.

## Quick Start

1. **Dataset Download (Optional)**:
   
   For the CrackForest dataset:
   ```powershell
   python scripts/download_crackforest.py
   ```
   
   For the Pascal VOC 2012 dataset, the system will automatically download on first run (requires `download: true` configuration). To manually trigger, you can directly run the benchmark script, and the system will automatically detect and download missing datasets.

2. **Run Benchmark Tests**:
   
   Using the CrackForest dataset:
   ```powershell
   python scripts/run_benchmark.py --config configs/crackforest_benchmark.yaml
   ```
   
   Using the Pascal VOC 2012 dataset:
   ```powershell
   python scripts/run_benchmark.py --config configs/voc_benchmark.yaml
   ```
   
   After the benchmark test completes, evaluation metrics for all models will be saved to the `reports/<run_name>/` directory:
   
   - `<model>_metrics.json`: Detailed evaluation metrics for individual models
   - `benchmark_summary.csv` / `benchmark_summary.json`: Comparison summary table for all models

3. **Parameter Configuration**:
   - `--device cuda`: Specify GPU execution (requires CUDA environment support)
   - `--skip-train`: Skip training/fine-tuning stages for all models, only perform inference evaluation
   - `--save-predictions`: Save prediction masks for each model as `.npy` format files

## Model Architecture Overview

| Type | Registered Name | Description |
| ---- | --------------- | ----------- |
| Feature + CRF | `classical_crf` | Handcrafted features + Random Forest + DenseCRF |
| CNN | `fcn_resnet50`, `deeplabv3_resnet50` | Torchvision semantic segmentation backbones, fine-tunable |
| Transformer | `segformer_b0` | HuggingFace SegFormer-B0 model |
| Diffusion Style | `random_walker` | DDP (Diffusion Model) |
| Hybrid | `hybrid_unet_transformer` | Custom CNN + Multi-head self-attention hybrid model |
| CNN-CRF | `cnn_crf` | CNN prediction + DenseCRF end-to-end combination |
| Any Model + CRF Post-processing | `crf_wrapper` | Wraps any registered model and appends DenseCRF post-processing |

> All models are registered via `segmentation_benchmark.evaluation.registry` and can be easily extended.

### CRF Post-processing Evaluation

This project supports adding DenseCRF post-processing evaluation to **any registered model**. CRF post-processing can improve the smoothness and accuracy of segmentation boundaries. In the configuration file, use the `crf_wrapper` builder to wrap any base model:

```yaml
models:
  # Base model
  - name: fcn_resnet50
    builder: fcn_resnet50
    params:
      pretrained: true
      finetune_epochs: 50
  
  # CRF post-processing version of the same model
  - name: fcn_resnet50_crf_post
    builder: crf_wrapper
    params:
      base_builder: fcn_resnet50  # Specify the base model to wrap
      base_params:               # Base model parameters
        pretrained: true
        finetune_epochs: 0       # Usually use pretrained model, no fine-tuning
      crf_params:                # CRF post-processing parameters
        iterations: 5            # CRF iteration count
        gaussian_sxy: 3          # Gaussian smoothing parameter
        bilateral_sxy: 80        # Bilateral filter spatial parameter
        bilateral_srgb: 13       # Bilateral filter color parameter
```

**CRF Post-processing Parameter Description**:
- `iterations`: CRF inference iteration count (default 5, more iterations may improve results but increase computation time)
- `gaussian_sxy`: Spatial standard deviation for Gaussian smoothing (default 3)
- `bilateral_sxy`: Spatial standard deviation for bilateral filtering (default 50-80)
- `bilateral_srgb`: Color standard deviation for bilateral filtering (default 13)
- `compat_gaussian`: Gaussian compatibility weight (default 3)
- `compat_bilateral`: Bilateral compatibility weight (default 10)

The current configuration file has added CRF post-processing evaluation versions for all major models (FCN, DeepLabV3, SegFormer, Hybrid UNet), and you can directly run benchmark tests for comparison.

### Automatic Checkpoint Management

This framework supports **intelligent checkpoint management**, automatically saving and loading trained models:

1. **Auto-save**: After training completes, models are automatically saved to the `artifacts/checkpoints/` directory
2. **Auto-load**: If the configuration is the same, previously trained models will be automatically loaded on the next run, skipping the training stage
3. **CRF Post-processing Auto-matching**: CRF post-processing versions will automatically find and use trained base model checkpoints

**How It Works**:
- Each checkpoint generates a unique hash based on model configuration (model name, number of classes, learning rate, training epochs, etc.)
- When configurations match exactly, the corresponding checkpoint is automatically loaded
- CRF wrapper intelligently finds matching base model checkpoints (ignoring the `finetune_epochs` parameter)

**Example**:
```yaml
# First run: Train and save checkpoint
- name: fcn_resnet50
  builder: fcn_resnet50
  params:
    pretrained: true
    finetune_epochs: 50  # Train for 50 epochs, automatically saved after training

# Second run: Automatically load checkpoint, skip training
- name: fcn_resnet50_crf_post
  builder: crf_wrapper
  params:
    base_builder: fcn_resnet50
    base_params:
      pretrained: true
      finetune_epochs: 0  # Automatically find and load trained checkpoint
```

**Manual Checkpoint Management**:
- All checkpoints are saved in the `artifacts/checkpoints/` directory
- File name format: `{builder}_{config_hash}.pth`
- You can force retraining by deleting checkpoint files

## Evaluation Metrics

This framework calculates and outputs the following evaluation metrics by default:

- Pixel Accuracy
- Mean IoU (Mean Intersection over Union)
- Mean Precision / Recall / F1
- Mean Dice
- Per-class IoU / Precision / Recall / F1 / Dice
- Confusion Matrix

The metric calculation implementation is located in `src/segmentation_benchmark/metrics/metrics.py`, and users can extend it according to research needs.

## Configuration File Description

Key parameter descriptions in the `configs/crackforest_benchmark.yaml` configuration file:

```yaml
dataset:
  root: data/crackforest  # Data directory
  download: true          # Auto-download if missing
  image_size: 256         # Unified resolution
  train_ratio: 0.6        # Training set ratio
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

To add new models, simply append corresponding entries to the `models` configuration list, and ensure the corresponding `builder` has been registered in the model registry.

## Extension Guide

This project supports various baseline extensions for further development and collaboration.

### Adding New Models

1. **Create Model File**: Create a new Python file in the `src/segmentation_benchmark/models/` directory (e.g., `my_model.py`)

2. **Implement Model Class**: Inherit from `BaseSegmenter` and implement necessary methods:
   ```python
   from ..evaluation.registry import register_segmenter
   from .base import BaseSegmenter
   
   @register_segmenter("my_model")
   class MySegmenter(BaseSegmenter):
       def __init__(self, num_classes: int = 2, **kwargs):
           super().__init__(num_classes=num_classes, name="MyModel")
           # Initialize model
       
       def prepare(self, train_dataset=None, val_dataset=None):
           # Optional: Train/fine-tune model
           pass
       
       def predict_batch(self, batch):
           # Required: Return (N, H, W) prediction masks
           pass
       
       def predict_logits(self, batch):
           # Optional: Return (N, C, H, W) logits (for CRF post-processing)
           pass
   ```

3. **Register Model**: Ensure the model module is added to the `_MODEL_MODULES` list in `src/segmentation_benchmark/evaluation/registry.py`:
   ```python
   _MODEL_MODULES = [
       # ... other modules
       "segmentation_benchmark.models.my_model",  # Add your model module
   ]
   ```

4. **Use in Configuration File**: Add model configuration in the YAML configuration file:
   ```yaml
   models:
     - name: my_model
       builder: my_model
       params:
         num_classes: 2
         # Other parameters...
   ```

5. **Add CRF Post-processing Version** (Optional): Add CRF post-processing evaluation for the new model:
   ```yaml
   - name: my_model_crf_post
     builder: crf_wrapper
     params:
       base_builder: my_model
       base_params:
         num_classes: 2
       crf_params:
         iterations: 5
         gaussian_sxy: 3
         bilateral_sxy: 80
         bilateral_srgb: 13
   ```

### Other Extensions
- **Replace Dataset**: Implement a new Dataset class and corresponding `create_dataloaders` factory function, and reference it in the configuration file.
- **Custom Metrics**: Extend the `SegmentationMetrics` or `MetricsAggregator` class in the `metrics` module to implement custom evaluation metric calculation logic.

## Testing

```powershell
pytest -q
```

- `tests/test_metrics.py`: Verify the correctness of evaluation metric calculations
- `tests/test_data.py`: Verify dataset loading and DataLoader data splitting functionality
- `tests/test_registry.py`: Ensure key models are correctly registered in the registry

## License and Citation

- The project code defaults to MIT License. Specific license information can be viewed or adjusted in `pyproject.toml`.
- When using the CrackForest dataset, please follow its non-commercial usage license and cite the literature provided by the original authors in relevant papers or research reports.

## Acknowledgments

This project thanks the CrackForest dataset authors and the open-source community (including PyTorch, Torchvision, Transformers, scikit-image, pydensecrf, etc.) for providing excellent tools and resource support.

This project aims to provide a systematic evaluation framework for image segmentation tasks, supporting comparative analysis of various model architectures.

---

# 图像分割集成套件

这是一个完整的图像分割评测项目，用于在数据集上比较多种类型的分割模型，并统一输出评估指标。我们希望通过该工作验证crf后处理技巧在图像分割后处理中的实际作用。本项目支持两个默认数据集的评测模块，分别是 **"crackforest"** 数据集和 **voc_2012** 数据集 ，支持自动下载、数据集拆分、模型训练/微调、推理、指标计算和结果归档。

## 功能亮点

- 支持的数据驱动基准：CRF 特征模型、CNN、Transformer、DDP（扩散模型）、混合 CNN-Transformer、CNN-CRF、任意模型 + CRF 后处理。
- 自动化流水线：读取配置 → 下载/加载数据 → 构建模型 → 评测 → 导出 JSON/CSV 结果。
- 多指标评估：Pixel Accuracy、mIoU、Precision、Recall、F1、Dice 等。
- 结构化代码：模块化的 `src/segmentation_benchmark` 包，便于扩展自定义模型或数据集。
- 单元测试覆盖基础组件（指标计算、数据管线、注册表）。

## 文件树

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

## 数据集说明

### CrackForest Dataset

- **数据集名称**：CrackForest Dataset（包含 118 张城市道路裂缝图像）
- **官方来源**：<https://github.com/cuilimeng/CrackForest-dataset>
- **使用许可**：本数据集仅限非商业科研用途，使用时请遵循项目 README 中的引用要求。
- **数据准备**：可通过执行脚本 `python scripts/download_crackforest.py` 手动下载，或在运行基准测试时自动下载。

### Pascal VOC 2012 Dataset

- **数据集名称**：Pascal VOC 2012（包含 21 个类别的语义分割标注）
- **官方来源**：<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>
- **使用许可**：Pascal VOC 数据集遵循其官方使用许可，通常允许用于学术研究。
- **数据准备**：在配置文件中设置 `download: true` 时，系统将自动通过 torchvision 下载数据集；数据集将保存至 `data/voc/VOCdevkit/VOC2012/` 目录。
- **注意事项**：该数据集下载网站常年不稳定，如果需要数据集请联系邮箱：wangfeiming@mail.nankai.edu.cn,我们会寻求办法将数据集公开！

> 默认配置会将数据集划分为 Train:Val:Test = 60% : 20% : 20%。可以通过 YAML 配置自定义。

## 安装

本项目要求 Python 3.10 环境。请注意，pydensecrf 需要自行编译安装。 **配置时注意降级Cython至2.X版本进行解译** 。
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
#安装开发依赖
pip install -e .[dev]
```

> Windows PowerShell 下请使用如上命令；其他平台请自行调整虚拟环境激活方式。

## 快速开始

1. **数据集下载（可选）**：
   
   对于 CrackForest 数据集：
   ```powershell
   python scripts/download_crackforest.py
   ```
   
   对于 Pascal VOC 2012 数据集，系统会在首次运行时自动下载（需配置 `download: true`）。如需手动触发，可直接运行基准测试脚本，系统将自动检测并下载缺失的数据集。

2. **执行基准测试**：
   
   使用 CrackForest 数据集：
   ```powershell
   python scripts/run_benchmark.py --config configs/crackforest_benchmark.yaml
   ```
   
   使用 Pascal VOC 2012 数据集：
   ```powershell
   python scripts/run_benchmark.py --config configs/voc_benchmark.yaml
   ```

   基准测试完成后，所有模型的评估指标将保存至 `reports/<run_name>/` 目录：

   - `<model>_metrics.json`：单个模型的详细评估指标
   - `benchmark_summary.csv` / `benchmark_summary.json`：所有模型的对比汇总表

3. **参数配置**：
   - `--device cuda`：指定使用 GPU 运行（需要 CUDA 环境支持）
   - `--skip-train`：跳过所有模型的训练/微调阶段，仅执行推理评估
   - `--save-predictions`：将每个模型的预测掩码保存为 `.npy` 格式文件

## 模型架构概览

| 类型 | 注册名称 | 描述 |
| ---- | -------- | ---- |
| 特征 + CRF | `classical_crf` | 手工特征 + Random Forest + DenseCRF |
| CNN | `fcn_resnet50`, `deeplabv3_resnet50` | Torchvision 语义分割骨干，可微调 |
| Transformer | `segformer_b0` | HuggingFace SegFormer-B0 模型 |
| Diffusion 风格 | `random_walker` | DDP（扩散模型） |
| Hybrid | `hybrid_unet_transformer` | 自定义 CNN + 多头自注意力混合模型 |
| CNN-CRF | `cnn_crf` | CNN 预测 + DenseCRF 端到端组合 |
| 任意模型 + CRF 后处理 | `crf_wrapper` | 包装任意注册模型并追加 DenseCRF 后处理 |

> 所有模型均通过 `segmentation_benchmark.evaluation.registry` 注册，可轻松扩展。

### CRF 后处理评估

本项目支持为**任意已注册的模型**添加 DenseCRF 后处理评估。CRF 后处理可以改善分割边界的平滑性和准确性。在配置文件中，使用 `crf_wrapper` builder 包装任何基础模型即可：

```yaml
models:
  # 基础模型
  - name: fcn_resnet50
    builder: fcn_resnet50
    params:
      pretrained: true
      finetune_epochs: 50
  
  # 同一模型的CRF后处理版本
  - name: fcn_resnet50_crf_post
    builder: crf_wrapper
    params:
      base_builder: fcn_resnet50  # 指定要包装的基础模型
      base_params:               # 基础模型的参数
        pretrained: true
        finetune_epochs: 0       # 通常使用预训练模型，不微调
      crf_params:                # CRF后处理参数
        iterations: 5            # CRF迭代次数
        gaussian_sxy: 3          # 高斯平滑参数
        bilateral_sxy: 80        # 双边滤波空间参数
        bilateral_srgb: 13       # 双边滤波颜色参数
```

**CRF 后处理参数说明**：
- `iterations`: CRF 推理迭代次数（默认 5，更多迭代可能提升效果但增加计算时间）
- `gaussian_sxy`: 高斯平滑的空间标准差（默认 3）
- `bilateral_sxy`: 双边滤波的空间标准差（默认 50-80）
- `bilateral_srgb`: 双边滤波的颜色标准差（默认 13）
- `compat_gaussian`: 高斯兼容性权重（默认 3）
- `compat_bilateral`: 双边兼容性权重（默认 10）

当前配置文件中已为所有主要模型（FCN、DeepLabV3、SegFormer、Hybrid UNet）添加了 CRF 后处理评估版本，可以直接运行基准测试进行对比。

### 自动 Checkpoint 管理

本框架支持**智能 checkpoint 管理**，可以自动保存和加载训练好的模型：

1. **自动保存**：训练完成后，模型会自动保存到 `artifacts/checkpoints/` 目录
2. **自动加载**：如果配置相同，下次运行时会自动加载之前训练好的模型，跳过训练阶段
3. **CRF 后处理自动匹配**：CRF 后处理版本会自动查找并使用训练好的基础模型 checkpoint

**工作原理**：
- 每个 checkpoint 根据模型配置（模型名称、类别数、学习率、训练轮数等）生成唯一哈希
- 配置完全匹配时自动加载对应的 checkpoint
- CRF wrapper 会智能查找匹配的基础模型 checkpoint（忽略 `finetune_epochs` 参数）

**示例**：
```yaml
# 第一次运行：训练并保存 checkpoint
- name: fcn_resnet50
  builder: fcn_resnet50
  params:
    pretrained: true
    finetune_epochs: 50  # 训练50轮，训练完成后自动保存

# 第二次运行：自动加载 checkpoint，跳过训练
- name: fcn_resnet50_crf_post
  builder: crf_wrapper
  params:
    base_builder: fcn_resnet50
    base_params:
      pretrained: true
      finetune_epochs: 0  # 自动查找并加载训练好的 checkpoint
```

**手动管理 checkpoint**：
- 所有 checkpoint 保存在 `artifacts/checkpoints/` 目录
- 文件名格式：`{builder}_{config_hash}.pth`
- 可以通过删除 checkpoint 文件来强制重新训练

## 评估指标

本框架默认计算并输出以下评估指标：

- Pixel Accuracy（像素准确率）
- Mean IoU（平均交并比）
- Mean Precision / Recall / F1（平均精确率、召回率、F1 分数）
- Mean Dice（平均 Dice 系数）
- Per-class IoU / Precision / Recall / F1 / Dice（各类别的交并比、精确率、召回率、F1 分数、Dice 系数）
- Confusion Matrix（混淆矩阵）

指标计算的实现位于 `src/segmentation_benchmark/metrics/metrics.py`，用户可根据研究需求进行扩展。

## 配置文件说明

`configs/crackforest_benchmark.yaml` 配置文件中的关键参数说明：

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

如需添加新的模型，只需在 `models` 配置列表中追加相应条目，并确保对应的 `builder` 已在模型注册表中完成注册。

## 扩展指南
该项目支持各种基线扩展，以便进一步开发和交流。

### 添加新模型

1. **创建模型文件**：在 `src/segmentation_benchmark/models/` 目录下创建新的 Python 文件（例如 `my_model.py`）

2. **实现模型类**：继承 `BaseSegmenter` 并实现必要的方法：
   ```python
   from ..evaluation.registry import register_segmenter
   from .base import BaseSegmenter
   
   @register_segmenter("my_model")
   class MySegmenter(BaseSegmenter):
       def __init__(self, num_classes: int = 2, **kwargs):
           super().__init__(num_classes=num_classes, name="MyModel")
           # 初始化模型
       
       def prepare(self, train_dataset=None, val_dataset=None):
           # 可选：训练/微调模型
           pass
       
       def predict_batch(self, batch):
           # 必须实现：返回 (N, H, W) 的预测掩码
           pass
       
       def predict_logits(self, batch):
           # 可选：返回 (N, C, H, W) 的 logits（用于CRF后处理）
           pass
   ```

3. **注册模型**：确保模型模块被添加到 `src/segmentation_benchmark/evaluation/registry.py` 的 `_MODEL_MODULES` 列表中：
   ```python
   _MODEL_MODULES = [
       # ... 其他模块
       "segmentation_benchmark.models.my_model",  # 添加你的模型模块
   ]
   ```

4. **在配置文件中使用**：在 YAML 配置文件中添加模型配置：
   ```yaml
   models:
     - name: my_model
       builder: my_model
       params:
         num_classes: 2
         # 其他参数...
   ```

5. **添加 CRF 后处理版本**（可选）：为新模型添加 CRF 后处理评估：
   ```yaml
   - name: my_model_crf_post
     builder: crf_wrapper
     params:
       base_builder: my_model
       base_params:
         num_classes: 2
       crf_params:
         iterations: 5
         gaussian_sxy: 3
         bilateral_sxy: 80
         bilateral_srgb: 13
   ```

### 其他扩展
- **替换数据集**：实现新的 Dataset 类以及相应的 `create_dataloaders` 工厂函数，并在配置文件中进行引用。
- **自定义指标**：在 `metrics` 模块中扩展 `SegmentationMetrics` 或 `MetricsAggregator` 类，实现自定义评估指标的计算逻辑。

## 测试

```powershell
pytest -q
```

- `tests/test_metrics.py`：验证评估指标计算的正确性
- `tests/test_data.py`：验证数据集加载与 DataLoader 数据拆分功能
- `tests/test_registry.py`：确保关键模型已在注册表中正确注册

## 许可与引用

- 项目代码默认采用 MIT License，具体许可信息可在 `pyproject.toml` 中查看或调整。
- 使用 CrackForest 数据集时，请遵循其非商业使用许可，并在相关论文或研究报告中引用原作者提供的文献。

## 致谢

本项目感谢 CrackForest 数据集作者以及开源社区（包括 PyTorch、Torchvision、Transformers、scikit-image、pydensecrf 等）提供的优秀工具和资源支持。

本项目旨在为图像分割任务提供系统化的评估框架，支持多种模型架构的对比分析。
