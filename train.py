import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.translated_fmnist import TranslatedFashionMNIST
from models.vit import ViT
from utils import AverageMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        default="./translated_data/translated_fmnist_train.pt",
    )
    parser.add_argument(
        "--val-data",
        default="./translated_data/translated_fmnist_test.pt",
    )

    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = choose_device()

    print(f"=> Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = TranslatedFashionMNIST(args.train_data)
    val_dataset = TranslatedFashionMNIST(args.val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")

        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            args,
        )

        val_acc = validate(
            val_loader,
            model,
            criterion,
            device,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, "model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(
                f"=> Best model saved at epoch {epoch + 1} "
                f"with acc {best_acc:.2f}%"
            )


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        acc1, = accuracy(outputs, labels, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f"Epoch: [{epoch + 1}][{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
            )


@torch.no_grad()
def validate(val_loader, model, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        acc1, = accuracy(outputs, labels, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

    print(
        f" * Validation Acc@1 {top1.avg:.3f}% "
        f"| Loss {losses.avg:.4f}"
    )
    return top1.avg


if __name__ == "__main__":
    main()
