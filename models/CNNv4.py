import torch
import torch.nn as nn
from torchvision import transforms as T

def transform(img):
    t = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=[.0, .0, .0], std=[255., 255., 255.]),
    ])
    return t(img)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            VGGBlock(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2), # 16 * 16
        )
        self.conv2 = nn.Sequential(
            VGGBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2), # 8 * 8
            
        )
        self.conv3 = nn.Sequential(
            VGGBlock(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2), # 4 * 4
        )
        self.conv4 = nn.Sequential(
            VGGBlock(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=2) # 1 * 1
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16*16*64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        self.linear2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(8*8*128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        self.linear3 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(4*4*256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self.linear4 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1*1*512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.linear1(x)

        x = self.conv2(x)
        x2 = self.linear2(x)

        x = self.conv3(x)
        x3 = self.linear3(x)

        x = self.conv4(x)
        x4 = self.linear4(x)

        x = torch.mean(torch.stack([x1, x2, x3, x4]), axis=0)

        return x


if __name__ == "__main__":
    import sys
    sys.path.append('.')

    from torch.utils.data import DataLoader
    from dataset import ObjectDataset
    from config import DatasetConfig
    
    model = Model()
    dataset = ObjectDataset(DatasetConfig(), transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0)

    for imgs, targets in dataloader:
        probs = model(imgs)
        print(probs)
        break
    
    