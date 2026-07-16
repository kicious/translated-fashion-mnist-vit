import os
import random

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import FashionMNIST


#原始图片大小为28x28
#这里将原始图片放到64x64的画布中

CANVAS_SIZE = 64
IMAGE_SIZE = 28

DATA_ROOT = "./data"
SAVE_ROOT = "./translated_data"

SEED = 42


def set_seed(seed):
    """
    固定随机种子，使实验结果可以复现。
    """

    random.seed(seed)
    torch.manual_seed(seed)


def generate_dataset(
    train,
    mode,
    save_path,
    seed,
):
    """
    生成位置变化后的FashionMNIST数据集。

    mode="random"：
    图片随机放在画布中，对应情况A。

    mode="center"：
    图片固定放在画布中央，对应情况B。
    """

    set_seed(seed)

    dataset = FashionMNIST(
        root=DATA_ROOT,
        train=train,
        download=True,
    )

    images = dataset.data
    labels = dataset.targets

    num_samples = len(images)

    new_images = torch.zeros(
        num_samples,
        1,
        CANVAS_SIZE,
        CANVAS_SIZE,
        dtype=torch.uint8,
    )

    positions = torch.zeros(
        num_samples,
        2,
        dtype=torch.long,
    )

    max_offset = CANVAS_SIZE - IMAGE_SIZE

    for i in range(num_samples):
        image = images[i]

        if mode == "random":
            top = random.randint(0, max_offset)
            left = random.randint(0, max_offset)

        elif mode == "center":
            top = max_offset // 2
            left = max_offset // 2

        else:
            raise ValueError(
                "mode必须是random或者center"
            )

        new_images[
            i,
            0,
            top:top + IMAGE_SIZE,
            left:left + IMAGE_SIZE,
        ] = image

        positions[i, 0] = top
        positions[i, 1] = left

        if (i + 1) % 5000 == 0:
            print(
                f"已经处理 {i + 1}/{num_samples} 张图片"
            )

    data = {
        "images": new_images,
        "labels": labels,
        "positions": positions,
        "canvas_size": CANVAS_SIZE,
        "mode": mode,
    }

    torch.save(
        data,
        save_path,
    )

    print(f"数据已经保存到：{save_path}")

    return new_images, labels, positions


def save_examples(
    images,
    labels,
    positions,
    save_path,
    title,
):
    """
    保存6张数据集样例，用于实验报告。
    """

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(9, 7),
    )

    axes = axes.reshape(-1)

    for i in range(6):
        axes[i].imshow(
            images[i, 0].numpy(),
            cmap="gray",
        )

        top = positions[i, 0].item()
        left = positions[i, 1].item()

        axes[i].set_title(
            f"label={labels[i].item()}\n"
            f"position=({top}, {left})"
        )

        axes[i].axis("off")

    figure.suptitle(title)
    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"样例图片已经保存到：{save_path}")


def main():
    os.makedirs(
        SAVE_ROOT,
        exist_ok=True,
    )

    #A训练集：图片随机平移

    a_train_images, a_train_labels, a_train_positions = (
        generate_dataset(
            train=True,
            mode="random",
            save_path="./translated_data/A_train.pt",
            seed=SEED,
        )
    )

    #A测试集：图片随机平移
    #使用不同的随机种子，使测试集位置不同

    generate_dataset(
        train=False,
        mode="random",
        save_path="./translated_data/A_test.pt",
        seed=SEED + 1,
    )

    #B训练集：图片固定在画布中央

    b_train_images, b_train_labels, b_train_positions = (
        generate_dataset(
            train=True,
            mode="center",
            save_path="./translated_data/B_train.pt",
            seed=SEED,
        )
    )

    #B测试集：图片固定在画布中央

    generate_dataset(
        train=False,
        mode="center",
        save_path="./translated_data/B_test.pt",
        seed=SEED,
    )

    save_examples(
        images=a_train_images,
        labels=a_train_labels,
        positions=a_train_positions,
        save_path="./translated_data/A_examples.png",
        title="Dataset A: Random Translation",
    )

    save_examples(
        images=b_train_images,
        labels=b_train_labels,
        positions=b_train_positions,
        save_path="./translated_data/B_examples.png",
        title="Dataset B: Centered Images",
    )

    print()
    print("所有数据已经生成完成：")
    print("A_train.pt：随机平移训练集")
    print("A_test.pt ：随机平移测试集")
    print("B_train.pt：固定居中训练集")
    print("B_test.pt ：固定居中测试集")


if __name__ == "__main__":
    main()