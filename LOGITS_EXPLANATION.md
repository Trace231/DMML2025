# 什么是"真实的 logits"？

## 核心概念

### 1. **Logits（逻辑值）** - 模型输出的原始分数

Logits 是模型最后一层输出的**原始分数**，形状为 `(N, C, H, W)`：
- `N`: 批次大小
- `C`: 类别数量（如2：背景和前景）
- `H, W`: 图像高度和宽度

**特点**：
- 每个像素对**每个类别**都有一个分数
- 分数是**连续的浮点数**（可以是正数、负数或零）
- 包含了模型对每个类别的**置信度信息**

### 2. **硬预测（Hard Prediction）** - 通过 argmax 得到的类别标签

硬预测是通过对 logits 取 argmax 得到的**整数类别标签**，形状为 `(N, H, W)`：
- 每个像素只有一个类别值（0, 1, 2等）
- 丢失了所有概率信息

## 具体例子

假设我们有一个像素，模型输出的 logits 如下：

### 场景1：使用真实的 logits ✅

```python
# 模型输出的真实 logits（形状: (1, 2, 256, 256)）
logits = np.array([
    [[5.2, 3.1, 2.8, ...],   # 背景类别的分数
     [2.1, 4.5, 3.2, ...]]   # 前景类别的分数
])

# 经过 softmax 转换为概率
probabilities = softmax(logits)
# 结果可能是：
# 背景: [0.95, 0.20, 0.40, ...]  # 第一个像素很确定是背景
# 前景: [0.05, 0.80, 0.60, ...]  # 第二个像素很确定是前景，第三个不确定

# CRF 可以看到：
# - 第一个像素：95% 确定是背景，5% 是前景（很确定）
# - 第二个像素：20% 是背景，80% 是前景（很确定）
# - 第三个像素：40% 是背景，60% 是前景（不确定，边界区域）
```

**CRF 的优势**：
- 可以看到**不确定的区域**（如第三个像素：40% vs 60%）
- 可以利用**空间一致性**来优化这些不确定区域
- 可以**平滑边界**，同时保留高置信度的预测

### 场景2：使用硬预测（硬 one-hot）❌

```python
# 从 logits 取 argmax 得到硬预测
hard_preds = np.argmax(logits, axis=1)
# 结果：[[0, 1, 1, ...]]  # 每个像素只有一个类别值

# 转换为硬 one-hot（_one_hot 方法）
one_hot = np.array([
    [[1.0, 0.0, 0.0, ...],   # 背景：第一个像素是1，其他是0
     [0.0, 1.0, 1.0, ...]]   # 前景：第二个和第三个像素是1，其他是0
])

# 经过 softmax 后（虽然已经是0和1了）
probabilities = softmax(one_hot)
# 结果：
# 背景: [1.0, 0.0, 0.0, ...]  # 要么是1，要么是0
# 前景: [0.0, 1.0, 1.0, ...]  # 要么是1，要么是0
```

**问题**：
- **丢失了不确定性信息**：第三个像素原本是 40% vs 60%（不确定），现在变成了 0% vs 100%（完全确定）
- **CRF 无法优化**：输入已经是极端分布（0或1），CRF 没有优化空间
- **可能破坏正确预测**：原本很确定的预测（95% vs 5%）被强制变成 100% vs 0%，CRF 可能会"过度优化"

## 代码中的体现

### 当前代码流程

```python
# 在 CrfWrappedSegmenter.predict_logits() 中：
def predict_logits(self, batch):
    logits = self.base.predict_logits(batch)  # 尝试获取真实 logits
    if logits is None:  # 如果模型没有实现 predict_logits()
        preds = self.base.predict_batch(batch)  # 获取硬预测
        logits = self._one_hot(preds)  # ❌ 转换为硬 one-hot
    return logits
```

### 好的模型（有真实 logits）

```python
# CNN 模型（cnn.py）
def predict_logits(self, batch):
    self.model.eval()
    with torch.no_grad():
        images = batch["image"].to(self.device)
        outputs = self.model(images)["out"]  # ✅ 返回真实的 logits
        return outputs.detach().cpu().numpy()
        # 形状: (N, C, H, W)，包含连续的分数值
```

### 不好的情况（只有硬预测）

```python
# 如果模型只实现了 predict_batch()，没有 predict_logits()
def predict_batch(self, batch):
    # ... 模型推理 ...
    return hard_predictions  # 形状: (N, H, W)，只有类别标签

# 然后 _one_hot() 会创建硬 one-hot：
def _one_hot(self, preds):
    # preds: [[0, 1, 1, ...]]  # 硬预测
    # 转换为: [[[1,0,0,...], [0,1,1,...]]]  # 硬 one-hot
    # ❌ 丢失了所有概率信息
```

## 为什么这很重要？

### CRF 的工作原理

CRF（条件随机场）是一个**概率图模型**，它需要：

1. **初始概率分布**：每个像素对每个类别的概率
2. **空间一致性约束**：相邻像素应该倾向于相同类别
3. **图像特征约束**：颜色相似的像素应该倾向于相同类别

### 使用真实 logits 的好处

```
真实 logits → softmax → 概率分布
  ↓
  - 保留不确定性信息
  - CRF 可以看到哪些区域不确定
  - CRF 可以优化这些不确定区域
  - 不会破坏已经很确定的预测
```

### 使用硬 one-hot 的问题

```
硬预测 → one-hot → softmax → 极端概率分布
  ↓
  - 丢失不确定性信息
  - 所有预测都是 0% 或 100%
  - CRF 没有优化空间
  - 可能破坏正确的预测
```

## 实际影响

从基准测试结果看：

| 模型 | 是否有真实 logits？ | CRF 后处理效果 |
|------|-------------------|---------------|
| fcn_resnet50 | ✅ 有 | mIoU: 0.726 → 0.508（下降30%） |
| deeplabv3_resnet50 | ✅ 有 | mIoU: 0.738 → 0.531（下降28%） |
| segformer_b0 | ✅ 有 | mIoU: 0.619 → 0.005（几乎失效） |

**注意**：即使有真实 logits，CRF 仍然可能降低性能，因为：
- CRF 参数可能不合适
- 深度模型已经足够好，CRF 可能过度优化
- CRF 的优化目标可能与模型训练目标不一致

## 总结

**"使用真实的 logits"** 意思是：
- ✅ 使用模型输出的**原始分数**（连续的浮点数）
- ✅ 保留**概率信息**和**不确定性**
- ❌ **不要**使用硬预测（整数类别标签）
- ❌ **不要**转换为硬 one-hot（只有0和1）

这样 CRF 才能看到哪些区域不确定，并利用空间一致性来优化这些区域。

