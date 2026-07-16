import torch
from torch.utils.data import Dataset


class TranslatedFashionMNIST(Dataset):
    """
    读取generate_data.py生成的.pt文件。
    """

    def __init__(self, data_path):
        super().__init__()

        data = torch.load(
            data_path,
            map_location="cpu",
            weights_only=True,
        )

        #原始像素范围是0到255
        #除以255之后转换到0到1

        self.images = data["images"].float() / 255.0
        self.labels = data["labels"].long()
        self.positions = data["positions"].long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        return img, label