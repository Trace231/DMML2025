# 预训练模型支持的标签和数据集

## 预训练模型统一存放位置与下载方式

- **统一目录**: 所有预训练权重会统一保存在仓库下的 `artifacts/pretrained/` 目录中
  - Torchvision 语义分割权重: `artifacts/pretrained/torchvision/`
  - HuggingFace / SegFormer 权重: `artifacts/pretrained/huggingface/`
- **自动行为**:
  - 跑 `fcn_resnet50` / `deeplabv3_resnet50` 时，代码会先去上述目录查找对应的权重文件；
  - 如果不存在，就会自动从官方网站下载到这个目录，再加载；
  - 跑 `segformer_b0` 时，会把模型和处理器都缓存到 `artifacts/pretrained/huggingface/`。
- **一键预下载所有预训练权重**（推荐在有网络的机器上先跑一遍）：

```bash
python scripts/download_pretrained.py
```

- 仅下载 torchvision 语义分割模型权重：

```bash
python scripts/download_pretrained.py --only torchvision
```

- 仅下载 SegFormer 相关权重：

```bash
python scripts/download_pretrained.py --only segformer
```

> 在离线的 HPC 环境中，可以在有网的机器上先运行上述脚本，再把整个 `artifacts/pretrained/` 目录拷到 HPC 相同位置使用。


## 模型预训练数据集总结

### 1. FCN ResNet50
- **预训练数据集**: **COCO** (Common Objects in Context)
- **类别数**: 21 类（包括背景）
- **COCO 语义分割类别**:
  - 0: 背景 (background)
  - 1: 人 (person)
  - 2: 自行车 (bicycle)
  - 3: 汽车 (car)
  - 4: 摩托车 (motorcycle)
  - 5: 飞机 (airplane)
  - 6: 公交车 (bus)
  - 7: 火车 (train)
  - 8: 卡车 (truck)
  - 9: 船 (boat)
  - 10: 交通灯 (traffic light)
  - 11: 消防栓 (fire hydrant)
  - 12: 停止标志 (stop sign)
  - 13: 停车计费器 (parking meter)
  - 14: 长椅 (bench)
  - 15: 鸟 (bird)
  - 16: 猫 (cat)
  - 17: 狗 (dog)
  - 18: 马 (horse)
  - 19: 羊 (sheep)
  - 20: 牛 (cow)
  - 21: 大象 (elephant)
  - 22: 熊 (bear)
  - 23: 斑马 (zebra)
  - 24: 长颈鹿 (giraffe)
  - 25: 背包 (backpack)
  - 26: 雨伞 (umbrella)
  - 27: 手提包 (handbag)
  - 28: 领带 (tie)
  - 29: 行李箱 (suitcase)
  - 30: 飞盘 (frisbee)
  - 31: 滑雪板 (skis)
  - 32: 雪板 (snowboard)
  - 33: 运动球 (sports ball)
  - 34: 风筝 (kite)
  - 35: 棒球棒 (baseball bat)
  - 36: 棒球手套 (baseball glove)
  - 37: 滑板 (skateboard)
  - 38: 冲浪板 (surfboard)
  - 39: 网球拍 (tennis racket)
  - 40: 瓶子 (bottle)
  - 41: 酒杯 (wine glass)
  - 42: 杯子 (cup)
  - 43: 叉子 (fork)
  - 44: 刀 (knife)
  - 45: 勺子 (spoon)
  - 46: 碗 (bowl)
  - 47: 香蕉 (banana)
  - 48: 苹果 (apple)
  - 49: 三明治 (sandwich)
  - 50: 橙子 (orange)
  - 51: 西兰花 (broccoli)
  - 52: 胡萝卜 (carrot)
  - 53: 热狗 (hot dog)
  - 54: 披萨 (pizza)
  - 55: 甜甜圈 (donut)
  - 56: 蛋糕 (cake)
  - 57: 椅子 (chair)
  - 58: 沙发 (couch)
  - 59: 盆栽 (potted plant)
  - 60: 床 (bed)
  - 61: 餐桌 (dining table)
  - 62: 厕所 (toilet)
  - 63: 电视 (tv)
  - 64: 笔记本电脑 (laptop)
  - 65: 鼠标 (mouse)
  - 66: 遥控器 (remote)
  - 67: 键盘 (keyboard)
  - 68: 手机 (cell phone)
  - 69: 微波炉 (microwave)
  - 70: 烤箱 (oven)
  - 71: 烤面包机 (toaster)
  - 72: 水槽 (sink)
  - 73: 冰箱 (refrigerator)
  - 74: 书 (book)
  - 75: 时钟 (clock)
  - 76: 花瓶 (vase)
  - 77: 剪刀 (scissors)
  - 78: 泰迪熊 (teddy bear)
  - 79: 吹风机 (hair drier)
  - 80: 牙刷 (toothbrush)

**注意**: torchvision 的 FCN ResNet50 实际上可能是在 **Pascal VOC** 上预训练的（21 类），而不是完整的 COCO。需要确认具体版本。

### 2. DeepLabV3 ResNet50
- **预训练数据集**: **COCO** 或 **Pascal VOC**
- **类别数**: 21 类（如果是在 VOC 上预训练）
- **Pascal VOC 类别**（21 类）:
  - 0: 背景 (background)
  - 1: 飞机 (aeroplane)
  - 2: 自行车 (bicycle)
  - 3: 鸟 (bird)
  - 4: 船 (boat)
  - 5: 瓶子 (bottle)
  - 6: 公交车 (bus)
  - 7: 汽车 (car)
  - 8: 猫 (cat)
  - 9: 椅子 (chair)
  - 10: 牛 (cow)
  - 11: 餐桌 (diningtable)
  - 12: 狗 (dog)
  - 13: 马 (horse)
  - 14: 摩托车 (motorbike)
  - 15: 人 (person)
  - 16: 盆栽 (pottedplant)
  - 17: 羊 (sheep)
  - 18: 沙发 (sofa)
  - 19: 火车 (train)
  - 20: 电视 (tvmonitor)

### 3. SegFormer B0
- **预训练数据集**: **ADE20K** (Scene Parsing Dataset)
- **模型名称**: `nvidia/segformer-b0-finetuned-ade-512-512`
- **类别数**: 150 类
- **ADE20K 主要类别**（150 类，包括）:
  - 建筑相关: 墙、地板、天花板、门、窗等
  - 家具: 床、椅子、桌子、沙发等
  - 交通工具: 汽车、公交车、火车等
  - 自然: 天空、山、水、树、草等
  - 人物: 人
  - 动物: 鸟、猫、狗等
  - 物品: 各种日常用品

## 推荐使用的数据集

### 对于 FCN ResNet50 和 DeepLabV3 ResNet50:
✅ **Pascal VOC 2012** - 完美匹配（21 类）
- 你的代码中已经有 `voc_benchmark.yaml` 配置
- 类别完全匹配预训练权重

✅ **COCO** - 如果模型是在 COCO 上预训练的
- 需要下载 COCO 数据集

### 对于 SegFormer B0:
✅ **ADE20K** - 完美匹配（150 类）
- 需要下载 ADE20K 数据集
- 类别完全匹配预训练权重

### 其他选项:
⚠️ **CrackForest** - 当前使用中（2 类：背景/裂缝）
- 所有模型都可以使用，但需要微调
- 预训练权重提供特征提取能力，但分类头需要重新训练
- 你的代码已经支持这个数据集

## 建议

1. **如果你想直接使用预训练权重（不微调）**:
   - 使用 **Pascal VOC 2012** 测试 FCN/DeepLabV3
   - 使用 **ADE20K** 测试 SegFormer

2. **如果你想微调模型**:
   - 可以使用任何数据集（如 CrackForest）
   - 预训练权重会提供更好的特征提取能力
   - 分类头会根据你的数据集类别数自动调整

3. **你的代码已经支持**:
   - ✅ Pascal VOC 2012（通过 `voc_benchmark.yaml`）
   - ✅ CrackForest（通过 `crackforest_benchmark.yaml`）
   - ❌ ADE20K（需要添加支持）

## 如何确认 torchvision 模型的预训练数据集

运行以下代码可以查看：
```python
import torchvision.models.segmentation as tv_seg
print(tv_seg.FCN_ResNet50_Weights.DEFAULT)
print(tv_seg.DeepLabV3_ResNet50_Weights.DEFAULT)
```

这会显示权重名称，通常包含数据集信息（如 "COCO" 或 "VOC"）。

