# CRF 优化测试指南

## 概述

`test_crf_optimizations.py` 脚本允许你测试新的 CRF 优化（自适应 CRF、边界感知 CRF 等），**无需重新训练模型**。它会：

1. 自动加载已保存的模型 checkpoint
2. 使用不同的 CRF 配置进行测试
3. 生成对比结果，不会覆盖原有模型

## 使用方法

### 基本用法

```bash
# 使用默认配置测试所有模型
python scripts/test_crf_optimizations.py

# 指定配置文件
python scripts/test_crf_optimizations.py --config configs/voc_benchmark.yaml

# 测试特定 epoch 的 checkpoint
python scripts/test_crf_optimizations.py --epoch 60

# 指定输出目录
python scripts/test_crf_optimizations.py --output reports/my_test

# 使用 CPU（如果默认是 GPU）
python scripts/test_crf_optimizations.py --device cpu
```

### 测试的 CRF 配置

脚本会自动测试以下 CRF 配置：

1. **baseline**: 标准 CRF（基准）
2. **adaptive_crf**: 自适应 CRF（根据图像大小和熵自动调整参数）
3. **adaptive_crf_full**: 自适应 CRF（所有特性启用，包括对比度）
4. **crf_more_iterations**: 标准 CRF，但迭代次数更多（10次）

### 输出结果

脚本会在输出目录生成以下文件：

- `crf_optimizations_results.csv`: 详细的测试结果（每个配置的指标）
- `crf_optimizations_results.json`: JSON 格式的详细结果
- `crf_optimizations_comparison.csv`: 按模型分组的对比统计

### 示例输出

```
================================================================================
CRF Optimization Test Results Summary
================================================================================

fcn_resnet50:
  Best mIoU: 0.7234 (adaptive_crf)
  Best Dice: 0.8412 (adaptive_crf)
  Configurations tested: 4

deeplabv3_resnet50:
  Best mIoU: 0.7456 (adaptive_crf_full)
  Best Dice: 0.8567 (adaptive_crf_full)
  Configurations tested: 4
```

## 工作原理

1. **不会重新训练**: 脚本只加载已保存的 checkpoint，不会触发训练
2. **不会覆盖模型**: 测试结果保存在独立的目录，不会影响原有模型
3. **自动匹配 checkpoint**: 脚本会自动找到匹配的 base model checkpoint
4. **支持 epoch checkpoint**: 可以测试特定 epoch 的模型性能

## 与原有评估脚本的区别

| 特性 | `evaluate_epoch_checkpoints.py` | `test_crf_optimizations.py` |
|------|--------------------------------|----------------------------|
| 用途 | 评估训练过程中的 epoch checkpoint | 测试不同的 CRF 优化配置 |
| 训练 | 如果需要会触发训练 | 只使用已有 checkpoint |
| CRF 配置 | 单一配置（通过 `--crf-params`） | 多个配置自动对比 |
| 输出 | 训练进度和 CRF 对比图 | CRF 优化对比结果 |

## 常见问题

### Q: 如果找不到 checkpoint 怎么办？

A: 确保模型已经训练过。可以运行：
```bash
python scripts/run_benchmark.py --config configs/voc_benchmark.yaml
```

### Q: 可以添加自定义 CRF 配置吗？

A: 可以！编辑 `test_crf_optimizations.py` 中的 `crf_configs` 列表，添加你的配置：

```python
{
    "name": "my_custom_crf",
    "description": "My custom CRF configuration",
    "params": {
        "use_adaptive_crf": False,
        "crf_params": {
            "iterations": 8,
            "bilateral_sxy": 100,
            # ... 其他参数
        },
    },
}
```

### Q: 测试结果保存在哪里？

A: 默认保存在 `reports/{run_name}/crf_optimizations/`，可以通过 `--output` 参数自定义。

### Q: 会影响原有的评估结果吗？

A: 不会。测试结果保存在独立目录，不会覆盖任何原有结果。

## 下一步

测试完成后，你可以：

1. 查看 `crf_optimizations_results.csv` 找到最佳配置
2. 在配置文件中使用最佳配置
3. 运行完整评估验证效果

例如，如果发现 `adaptive_crf` 效果最好，可以在配置文件中使用：

```yaml
- name: fcn_resnet50_adaptive_crf
  builder: cnn_crf
  params:
    base_model: fcn_resnet50
    pretrained: true
    finetune_epochs: 0
    use_adaptive_crf: true
    adaptive_crf_config:
      scale_by_image_size: true
      scale_by_entropy: true
```

