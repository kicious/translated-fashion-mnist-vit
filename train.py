import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vit import ViT
from datasets.translated_fmnist import TranslatedFashionMNIST
from utils import AverageMeter, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Translated FashionMNIST Training')

    # Dataset
    parser.add_argument('--train-data',type=str,default='./translated_data/translated_fashion_mnist_train.pt',help='path to train data')
    parser.add_argument('--val-data',type=str,default='./translated_data/translated_fashion_mnist_test.pt',help='path to val data')

    #Model
    parser.add_argument('--img-size',type=int,default=128,help='input image size')
    parser.add_argument('--patch-size',type=int,default=16,help='patch size')
    parser.add_argument('--embed-dim',type=int,default=128,help='enbedding dimension')
    parser.add_argument('--depth',type=int,default=4,help='transformer encoder')
    parser.add_argument('--num-heads',type=int,default=4,help='number of attention heads')

    # Optimization
    parser.add_argument('--epoches',default=15,type=int,help='number of total epochs to run')
    parser.add_argument('--batch-size',default=128,type=int,help='mini-batch size')
    parser.add_argument('--lr','--learning-rate',default=1e-3,type=float,help='initial learning rate',dest='lr')
    parser.add_argument('--weight-decay',default=1e-4,type=float,help='weight decay')

    #Misc
    parser.add_argument('--print-freq',default=50,type=int,help='print frequency')
    parser.add_argument('--save-dir',default='./checkpoints',type=str,help='path to save checkpoints')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"=> Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    #1.Dataset & DataLoader
    print("=> Loading datasets...")
    train_dataset = TranslatedFashionMNIST(args.train_data)
    val_dataset = TranslatedFashionMNIST(args.val_data)
    
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle = True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle = False,num_workers=4,pin_memory=True)

    #2.Model
    print("=>Building model...")
    model = ViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)

    #3.Criterion & Optimizer
    criterion = nn.CrossEntroyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    #4.Training Loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        train(train_loader,model,criterion,optimizer,epoch,device,args)

        val_acc = validate(val_loader,model,criterion,device,args)

        #Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir,'model_best.pth')
            torch.save(model.state_dict(),save_path)
            print("=> Best model saved at epoch {epoch} with acc {best_acc:.2f}%")

def train(train_loader,model,criterion,optimizer,epoch,device,args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i,(images,labels) in enumerate(train_loader):
        images,labels = images.to(device),labels.to(device)

        #Compute output
        outputs = model(images)
        loss = criterion(outputs,labels)

        #Measure accuracy and record loss
        acc1, = accuracy(outputs,labels,topk=(1,))
        losses.update(loss.item(),images.size(0))
        top1.update(acc1.item(),images.size(0))

        #Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()

        if i% args.print_freq == 0:
            print(f"Epoch:[{epoch}][{i}/{len(train_loader)}]\t"
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'  )
            
def validate(val_loader,model,criterion,device,args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (images,labels) in enumerate(val_loader):
            images,labels = images.to(device),labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)

            acc1, = accuracy(outputs,labels,topk=(1,))
            losses.update(loss.item(),images.size(0))
            top1.update(acc1.item(),images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        print(f'* Validation Acc@1 {top1.avg:.3f}% | Loss {losses.avg:.4f}')

if __name__ == '__main__':
    main()
             
