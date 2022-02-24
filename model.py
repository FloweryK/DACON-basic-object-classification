import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(32*32*3, 10),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = torch.flatten(x)
        x = self.layer1(x)
        return x


if __name__ == "__main__":
    import cv2

    model = Model()
    x = cv2.imread(os.path.join('data', 'train', 'airplane', '0000.jpg'))
    x = torch.tensor(x, dtype=torch.float)

    x = model(x)
    print(x)
    