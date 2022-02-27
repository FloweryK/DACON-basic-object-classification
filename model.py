import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # nn.Flatten(start_dim=1),    # to reshape with considering batch size
            # nn.Linear(32*32*3, 10),
            # nn.Softmax(dim=0)           # nn.CrossEntropyLoss already includes softmax
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Flatten(start_dim=1),
            nn.Linear(57600, 10)
        )
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        return x


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import ObjectDataset
    from config import DatasetConfig
    
    model = Model()
    dataset = ObjectDataset(DatasetConfig())
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0)

    for imgs, targets in dataloader:
        probs = model(imgs)
        print(probs)
        break
    
    