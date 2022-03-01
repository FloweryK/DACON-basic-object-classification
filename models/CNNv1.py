import torch.nn as nn
from torchvision import transforms as T

def transform(img):
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[.0, .0, .0], std=[255., 255., 255.]),
    ])
    return t(img)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),

            nn.AvgPool2d(kernel_size=2),
        )

        self.linear1 = nn.Sequential(    
            # Flatten
            nn.Flatten(start_dim=1),    # to reshape with considering batch size
            nn.Linear(512, 256),
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
    
    