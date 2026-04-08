# 肺结节检测与分类系统

基于 DINOv2 自监督学习的肺结节检测与分类项目。

## 项目简介

- **数据集**: LUNA16
- **模型**: DINOv2 ViT-L/14 + 双头任务网络
- **任务**: 结节分类 + 边界框回归

## 文件说明

| 文件 | 功能 |
|------|------|
| `preprocess_luna16.py` | 数据预处理（3D CT → 2D PNG 切片） |
| `feature_extraction.py` | DINOv2 特征提取 |
| `train_class_dete.py` | 模型训练与评估 |
| `dataset.py` | 数据集加载（含 3D 上下文编码） |
| `model.py` | 双头模型定义 |
| `config.py` | 配置文件 |

## 快速开始

```bash
# 1. 预处理数据
python preprocess_luna16.py

# 2. 提取特征
python feature_extraction.py

# 3. 训练模型
python train_class_dete.py
```

## 输出文件

- `train/val/test_features.npy` - DINOv2 特征
- `best_model.pth` - 训练好的模型
- `final_report.json` - 评估结果

## 环境要求

- Python 3.8+
- PyTorch
- DINOv2
- scikit-learn
