import torch
from torch.utils.data import Dataset


class TranslatedFashionMNIST(Dataset):
    #python类的构造函数，负责在对象创建时完成初始化工作
    def __init__(self, data_path):
        super(TranslatedFashionMNIST, self).__init__()

        # 加载预生成的 .pt 文件，保存图像，标签，位置信息
        data = torch.load(data_path)
        self.images = data['images']
        self.labels = data['labels']
        self.positions = data['positions']

    #数据集的样本数量
    def __len__(self):
        return len(self.labels)

    #根据索引idx取出一个样本
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # 如果需要位置信息作为辅助任务，可以 return pos
        return img, label