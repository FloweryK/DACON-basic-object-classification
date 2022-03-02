import torch.nn as nn
from torchvision import transforms as T

def transform(img):
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[.0, .0, .0], std=[255., 255., 255.]),
    ])
    return t(img)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=5, padding=3),
            ConvBlock(64, 64, kernel_size=5, padding=3),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=5, padding=3),
            ConvBlock(128, 128, kernel_size=5, padding=3),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=5, padding=3),
            ConvBlock(256, 256, kernel_size=5, padding=3),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=5, padding=3),
            ConvBlock(512, 512, kernel_size=5, padding=3),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2)
        )

        self.linear1 = nn.Sequential(    
            # Flatten
            nn.Flatten(start_dim=1),    # to reshape with considering batch size
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.linear1(x)
        x = self.linear2(x)

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
    
    