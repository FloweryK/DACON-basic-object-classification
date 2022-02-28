import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # Conv 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
            # Conv 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            
            # Conv 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),

            # Conv 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),

            nn.AvgPool2d(kernel_size=2),
            
            # Flatten
            nn.Flatten(start_dim=1),    # to reshape with considering batch size
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 10)
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
    
    