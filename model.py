import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # CNN Block 
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.35),

            # CNN Block 
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.35),

            # CNN Block 
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.35),

            # Flatten
            nn.Flatten(start_dim=1),    # to reshape with considering batch size
            
            # Linear Block
            nn.Linear(16384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Linear Block
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Block
            nn.Linear(256, 10),
            # nn.Softmax(dim=0)           # nn.CrossEntropyLoss already includes softmax
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
    
    