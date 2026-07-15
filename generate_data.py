import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
from torchvision import transforms


def create_translated_fashion_mnist( root_dir="./data", save_dir="./translated_data", canvas_size=128,is_train=True):
    """
    生成位置可变的 FashionMNIST 并保存。
    """
    # 加载原始的FashionMNIST（28×28）
    origin_dataset = torchvision.datasets.FashionMNIST( root=root_dir,train=is_train,download=True,transform=transforms.ToTensor())
    #记录下样本数量，小图像大小
    num_samples = len(origin_dataset)
    img_size = 28
    # 预分配张量(提前创建内存空间)，图像形状：(N, 1, canvas_size, canvas_size)
    translated_images = torch.zeros((num_samples, 1, canvas_size, canvas_size),dtype=torch.float32)
    # 初始化标签    
    labels = torch.zeros(num_samples, dtype=torch.long)
    # 保存每张图片左上角的 (y, x) 坐标
    positions = torch.zeros( (num_samples, 2),dtype=torch.long,)
 
    #真正开始生成数据
    split_name = "训练集" if is_train else "测试集"
    print(f"正在生成 {split_name}（共 {num_samples} 个样本）...")
    for i in range(num_samples):
        # img 的形状是 (1, 28, 28)
        img, label = origin_dataset[i]
        # 生成随机位置，确保图片完全在画布内
        max_y = canvas_size - img_size
        max_x = canvas_size - img_size
        y = np.random.randint(0,max_y + 1,)
        x = np.random.randint( 0, max_x + 1,)
        # 将原图放入大画布
        translated_images[i,0, y:y + img_size, x:x + img_size,] = img[0]
        # 保存标签
        labels[i] = label
        # 保存位置
        positions[i] = torch.tensor([y,x])
        # 打印进度
        if (i + 1) % 5000 == 0:
            print(f"已完成 {i + 1}/{num_samples}")
                
    # 保存数据
    # 创建保存目录
    os.makedirs(save_dir,exist_ok=True)
    # 训练集与测试集使用不同文件名
    filename = (
        f"translated_fmnist_"
        f"{'train' if is_train else 'test'}.pt"
    )
    save_path = os.path.join(save_dir, filename)
    #将图片、标签、位置保存到同一个 .pt 文件
    torch.save(
        {
            "images": translated_images,
            "labels": labels,
            "positions": positions,
        },
        save_path,
    )
    print( f"数据已保存至：{save_path}\n")

    # 返回数据
    return (translated_images,labels,positions,)
        
    


def show_examples(images, labels, positions,count=6):
    """
    显示若干生成后的样本。
    """
    count = min( count,len(images))
    fig, axes = plt.subplots(2,3,figsize=(10,7))
    axes = axes.reshape(-1)

    for i in range(count):
        axes[i].imshow(images[i, 0],cmap="gray",)
        y, x = positions[i].tolist()
       # 使用红色方框标记原图位置
        rect = patches.Rectangle(
            (x, y),
            28,
            28,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        axes[i].add_patch(rect)
        axes[i].set_title(
            f"label={int(labels[i])}, "
            f"pos=({y}, {x})"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


#只有该文件被直接运行时才会执行以下代码
if __name__ == "__main__":
    # 固定随机种子，便于复现实验
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成训练集
    train_images, train_labels, train_positions = (
        create_translated_fashion_mnist(
            canvas_size=128,
            is_train=True,
        )
    )

    # 生成测试集
    test_images, test_labels, test_positions = (
        create_translated_fashion_mnist(
            canvas_size=128,
            is_train=False,
        )
    )

    # 显示训练集中的前 6 个样本
    show_examples(
        train_images,
        train_labels,
        train_positions,
    )