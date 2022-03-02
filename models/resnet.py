import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18

def transform(img):
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[.0, .0, .0], std=[255., 255., 255.]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return t(img)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.fc = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)

        return x