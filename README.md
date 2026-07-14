# Translated FashionMNIST ViT

基于位置可变 FashionMNIST 数据集的 Vision Transformer 分类实验。

## 项目结构

- `generate_data.py`：生成位置可变的 FashionMNIST 数据
- `datasets/`：数据集读取代码
- `models/`：ViT 模型代码
- `utils.py`：训练工具函数
- `train.py`：训练和验证入口
- `check_install.py`：检查 PyTorch 和加速设备

## 安装普通依赖

```bash
python -m pip install -r requirements.txt
