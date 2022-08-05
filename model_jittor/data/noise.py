import jittor as jt
from jittor.dataset import Dataset


class NoiseDataset(Dataset):
    def __init__(self, length=50000, image_size=32, channel=3):
        super().__init__()
        self.image_size = image_size
        self.channel = channel
        self.set_attrs(total_len=length)

    def __getitem__(self, idx):
        return jt.randn(self.channel, self.image_size, self.image_size), idx


if __name__ == "__main__":
    dataloader = NoiseDataset(
        length=100, 
        image_size=32, 
        channel=3,
    ).set_attrs(
        batch_size=20, shuffle=True,
    )
    for imgs, ids in dataloader:
        print(imgs.shape, ids.data)