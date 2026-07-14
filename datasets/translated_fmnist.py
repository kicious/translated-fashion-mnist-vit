import torch
from torch.utils.data import Dataset


class TranslatedFashionMNIST(Dataset):
    """
    读取 generate_data.py 生成的 .pt 文件。
    """

    def __init__(self, data_path):
        super().__init__()

        data = torch.load(
            data_path,
            map_location="cpu",
            weights_only=True,
        )

        self.images = data["images"].float()
        self.labels = data["labels"].long()
        self.positions = data["positions"].long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        return img, label