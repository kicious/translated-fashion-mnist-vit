import argparse
import csv
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.translated_fmnist import TranslatedFashionMNIST
from models.vit import ViT


#四种实验设置
SETTINGS = {
    1: {
        "train_file": "A_train.pt",
        "test_file": "A_test.pt",
        "description": "A训练集 -> A测试集",
    },

    2: {
        "train_file": "B_train.pt",
        "test_file": "B_test.pt",
        "description": "B训练集 -> B测试集",
    },

    3: {
        "train_file": "A_train.pt",
        "test_file": "B_test.pt",
        "description": "A训练集 -> B测试集",
    },

    4: {
        "train_file": "B_train.pt",
        "test_file": "A_test.pt",
        "description": "B训练集 -> A测试集",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--setting",type=int,choices=[1, 2, 3, 4], default=1,)
    parser.add_argument( "--data-dir",type=str,default="./translated_data",)
    parser.add_argument( "--result-dir",type=str,default="./results",)
    parser.add_argument("--img-size",type=int,default=64,)
    parser.add_argument("--patch-size",type=int,choices=[4, 8, 16],default=16,)
    parser.add_argument("--embed-dim", type=int,default=128,)
    parser.add_argument("--depth",type=int,default=4,)
    parser.add_argument("--num-heads", type=int,default=4,)
    parser.add_argument("--mlp-ratio",type=float,default=4.0,)
    parser.add_argument("--batch-size", type=int, default=128,)
    parser.add_argument( "--epochs",type=int,default=15,)
    parser.add_argument("--lr",type=float,default=0.001,)
    parser.add_argument("--weight-decay", type=float, default=0.0001,)
    parser.add_argument("--num-workers",type=int,default=0,)
    parser.add_argument("--seed",type=int,default=42,)
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """
    固定随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    自动选择训练设备。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def get_data_paths(setting,data_dir,):
    """
    根据setting选择训练集和测试集。
    """
    setting_info = SETTINGS[setting]
    train_path = os.path.join(data_dir,setting_info["train_file"],)
    test_path = os.path.join(data_dir,setting_info["test_file"],)
    return train_path, test_path


def train_one_epoch(model,train_loader,criterion,optimizer,device,):
    """
    完成一个epoch的训练。
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        #清空上一次反向传播留下的梯度
        optimizer.zero_grad()
        #将图片输入模型，得到预测结果
        outputs = model(images)
        #计算预测结果与真实标签之间的loss
        loss = criterion(outputs,labels)
        #反向传播，计算梯度
        loss.backward()
        #根据梯度更新模型参数
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
    
    train_loss = (total_loss / total_samples)
    train_acc = (total_correct/ total_samples* 100)
    return train_loss, train_acc


def test_one_epoch(model,test_loader,criterion,device,):
    """
    在测试集上测试模型。
    """
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            batch_size = labels.size(0)
            total_loss += (loss.item() * batch_size)
            predictions = outputs.argmax(dim=1,)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size
    test_loss = ( total_loss / total_samples)
    test_acc = (total_correct /total_samples*100)
    return test_loss, test_acc

def create_result_dir(setting,patch_size,result_root):
    """
    为本次训练建立单独的结果文件夹。
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (f"setting{setting}_patch{patch_size}_{current_time}")
    result_dir = os.path.join(result_root,experiment_name)
    os.makedirs(result_dir,exist_ok=True)

    return result_dir


def save_config(args,result_dir,train_path,test_path,device,):
    """
    保存本次训练使用的参数。
    """
    config = vars(args).copy()
    config["description"] = (SETTINGS[args.setting]["description"])
    config["train_path"] = train_path
    config["test_path"] = test_path
    config["device"] = str(device)
    config_path = os.path.join(result_dir,"config.json",)
    with open(config_path,"w",encoding="utf-8",) as file:
        json.dump(config,file,ensure_ascii=False,indent=4,)


def create_metrics_file(result_dir):
    """
    创建保存每个epoch结果的csv文件。
    """
    metrics_path = os.path.join(result_dir,"metrics.csv",)
    with open(metrics_path,"w",newline="",encoding="utf-8-sig",)as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","train_loss","train_acc","test_loss","test_acc",])
    return metrics_path


def save_one_epoch(metrics_path,epoch,train_loss,train_acc,test_loss,test_acc,):
    """
    将一个epoch的训练结果写入csv文件。
    """
    with open(metrics_path,"a",newline="",encoding="utf-8-sig",) as file:
        writer = csv.writer(file)
        writer.writerow([epoch,train_loss,train_acc,test_loss,test_acc])

def draw_curves( history,result_dir,):
    """
    画出loss和accuracy随epoch变化的曲线。
    """
    epochs = history["epoch"]
    #画loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs,history["train_loss"],label="Train Loss",)
    plt.plot(epochs,history["test_loss"],label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    loss_curve_path = os.path.join(result_dir,"loss_curve.png",)
    plt.savefig(loss_curve_path,dpi=200,bbox_inches="tight")
    plt.close()

    #画accuracy曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs ,history["train_acc"],label="Train Accuracy",)
    plt.plot(epochs,history["test_acc"],label="Test Accuracy",)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()
    accuracy_curve_path = os.path.join(result_dir,"accuracy_curve.png",)
    plt.savefig(accuracy_curve_path,dpi=200,bbox_inches="tight",)
    plt.close()


def save_summary(args,result_dir,best_epoch,best_acc,history,):
    """
    保存本次实验的最终结果。
    """
    summary = {
        "setting": args.setting,
        "description": (
            SETTINGS[args.setting]["description"]
        ),
        "patch_size": args.patch_size,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "final_train_loss": (
            history["train_loss"][-1]
        ),
        "final_train_acc": (
            history["train_acc"][-1]
        ),
        "final_test_loss": (
            history["test_loss"][-1]
        ),
        "final_test_acc": (
            history["test_acc"][-1]
        ),
    }
    summary_path = os.path.join(result_dir,"summary.json",)
    with open(summary_path,"w", encoding="utf-8", ) as file:
        json.dump(summary,file,ensure_ascii=False,indent=4,)
            

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.img_size % args.patch_size != 0:
        raise ValueError("img_size必须能够被patch_size整除")
    device = get_device()
    train_path, test_path = get_data_paths(setting=args.setting,data_dir=args.data_dir,)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集：{train_path}\n请先运行python generate_data.py")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集：{test_path}\n请先运行python generate_data.py")
    result_dir = create_result_dir(setting=args.setting,patch_size=args.patch_size,result_root=args.result_dir,)
    
    print("=" * 70)
    print(f"setting：{args.setting}")
    print(
        f"实验设置："
        f"{SETTINGS[args.setting]['description']}"
    )
    print(f"训练集：{train_path}")
    print(f"测试集：{test_path}")
    print(f"patch_size：{args.patch_size}")
    print(f"batch_size：{args.batch_size}")
    print(f"epochs：{args.epochs}")
    print(f"learning rate：{args.lr}")
    print(f"训练设备：{device}")
    print(f"结果目录：{result_dir}")
    print("=" * 70)

    train_dataset = TranslatedFashionMNIST(train_path)
    test_dataset = TranslatedFashionMNIST(test_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,)

    model = ViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,)
    save_config(args=args,result_dir=result_dir,train_path=train_path,test_path=test_path,device=device,)
    metrics_path = create_metrics_file(result_dir)
    history = {"epoch": [],"train_loss": [],"train_acc": [],"test_loss": [],"test_acc": [],}

    best_acc = 0
    best_epoch = 0
    for epoch in range(1,args.epochs + 1):
        train_loss, train_acc = (
            train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )
        test_loss, test_acc = (
            test_one_epoch(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
            )
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
          
        save_one_epoch(
            metrics_path=metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

            best_model_path = os.path.join(
                result_dir,
                "best_model.pth",
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "test_acc": test_acc,
                    "args": vars(args),
                },
                best_model_path,
            )

            print(
                f"保存最佳模型："
                f"epoch={epoch}，"
                f"test_acc={test_acc:.2f}%"
            )

        draw_curves(
            history=history,
            result_dir=result_dir,
        )

    final_model_path = os.path.join(
        result_dir,
        "final_model.pth",
    )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "args": vars(args),
        },
        final_model_path,
    )

    save_summary(
        args=args,
        result_dir=result_dir,
        best_epoch=best_epoch,
        best_acc=best_acc,
        history=history,
    )

    print()
    print("训练完成")
    print(f"最佳epoch：{best_epoch}")
    print(f"最佳测试准确率：{best_acc:.2f}%")
    print(f"实验结果保存在：{result_dir}")


if __name__ == "__main__":
    main()