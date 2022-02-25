import os
import torch
import torch.nn as nn
from torchvision import transforms as T

def t(img):
    transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[113.86283869, 122.93301916, 125.26884795], std=[66.46033586, 61.90929531, 62.5340599])
    ])
    return transform(img)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(32*32*3, 10),
            # nn.Softmax(dim=0) # nn.CrossEntropyLoss already includes softmax
        )
    
    def forward(self, x):
        x = t(x)
        x = torch.flatten(x)
        x = self.layer1(x)
        return x


if __name__ == "__main__":
    import cv2

    model = Model()
    x = cv2.imread(os.path.join('data', 'train', 'airplane', '0000.jpg'))

    x = model(x)
    print(x)
    