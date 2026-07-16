# Translated FashionMNIST Vision Transformer

本项目使用 PyTorch 实现一个基于 Vision Transformer（ViT）的位置可变 FashionMNIST 图像分类系统。

项目的核心目标是研究：

> 当服装图像在画布中的位置发生变化时，使用绝对位置编码的 Vision Transformer 能否正确完成图像分类？

原始 FashionMNIST 图像大小为 `28 × 28`。本项目将原始图像放置到更大的 `64 × 64` 黑色画布中，并构造两类数据集：

- **数据集 A：随机平移数据集**  
  每张 `28 × 28` 图像随机出现在 `64 × 64` 画布中的不同位置。

- **数据集 B：固定居中数据集**  
  每张 `28 × 28` 图像都固定放置在 `64 × 64` 画布中央。

我们使用带有可学习绝对位置编码的 Vision Transformer 对这些图像进行分类，并比较不同训练集、测试集组合下的分类结果。

---

## 1. 项目研究内容

本项目主要完成以下工作：

1. 下载原始 FashionMNIST 数据集；
2. 将原始 `28 × 28` 图像扩展到 `64 × 64` 画布；
3. 生成随机平移数据集 A 和固定居中数据集 B；
4. 实现基于绝对位置编码的 Vision Transformer；
5. 使用四种训练集和测试集组合进行实验；
6. 记录每个 epoch 的训练损失、测试损失、训练准确率和测试准确率；
7. 保存训练曲线、实验参数、最佳模型和最终实验结果；
8. 比较图像位置变化对 ViT 分类性能的影响；
9. 尝试比较不同 patch size 对模型性能的影响。

---

## 2. 四种实验设置

项目使用以下四种实验设置：

| Setting | 训练集 | 测试集 | 实验含义 |
|---:|---|---|---|
| 1 | A 随机平移训练集 | A 随机平移测试集 | 在随机位置图像上训练和测试 |
| 2 | B 固定居中训练集 | B 固定居中测试集 | 在固定居中图像上训练和测试 |
| 3 | A 随机平移训练集 | B 固定居中测试集 | 随机位置训练后，测试居中图像 |
| 4 | B 固定居中训练集 | A 随机平移测试集 | 居中训练后，测试随机位置图像 |

四种 setting 使用同一个 ViT 模型和同一套训练代码，只有训练集和测试集的组合不同。

程序内部的数据选择关系如下：

```text
Setting 1：A_train.pt → A_test.pt
Setting 2：B_train.pt → B_test.pt
Setting 3：A_train.pt → B_test.pt
Setting 4：B_train.pt → A_test.pt
```
---

## 3. 项目结构
translated-fashion-mnist-vit/
│
├── data/
│   └── FashionMNIST/
│       └── 原始 FashionMNIST 数据
│
├── translated_data/
│   ├── A_train.pt
│   ├── A_test.pt
│   ├── B_train.pt
│   ├── B_test.pt
│   ├── A_examples.png
│   └── B_examples.png
│
├── datasets/
│   ├── __init__.py
│   └── translated_fmnist.py
│
├── models/
│   ├── __init__.py
│   └── vit.py
│
├── results/
│   └── 每次训练自动创建一个单独的实验目录
│
├── generate_data.py
├── train.py
├── run_all.py
├── utils.py
├── check_install.py
├── requirements.txt
├── README.md
└── .gitignore

---

## 4. 各文件作用

### `generate_data.py`
负责生成实验所需的数据集。  
它会读取原始 FashionMNIST，并生成：

- `translated_data/A_train.pt`
- `translated_data/A_test.pt`
- `translated_data/B_train.pt`
- `translated_data/B_test.pt`

同时还会生成：

- `translated_data/A_examples.png`
- `translated_data/B_examples.png`

这两张图片用于观察 A、B 数据集的区别，也可以放入实验报告。

### `datasets/translated_fmnist.py`
定义自定义 PyTorch 数据集 `TranslatedFashionMNIST`。  
主要功能：

- 读取 `.pt` 数据文件；
- 取得图像和标签；
- 将像素从 0～255 转换到 0～1；
- 为 `DataLoader` 提供数据。

每次返回：`image, label`

### `models/vit.py`
定义 Vision Transformer 模型。  

### train.py
负责运行一组实验。
主要完成：
根据 --setting 选择训练集和测试集；
创建 DataLoader；
创建 ViT 模型；
使用交叉熵损失函数；
使用 AdamW 优化器；
完成模型训练；
每个 epoch 在测试集上测试；
保存每个 epoch 的 loss 和 accuracy；
绘制训练曲线；
保存最佳模型和最终模型

### run_all.py
负责自动依次运行四种 setting。
它会按照以下顺序运行：
Setting 1
Setting 2
Setting 3
Setting 4
每一个 setting 都会重新创建并训练一个新的 ViT 模型。
不同 setting 之间不会共用模型参数。

### check_install.py
check_install.py
用于检查：
PyTorch 是否安装成功；
Torchvision 是否安装成功；
CUDA 是否可用；
当前最终使用的是 CUDA、MPS 还是 CPU。
在开始训练前建议先运行这个文件。

### utils.py
用于保存训练过程中可能使用的辅助函数。

