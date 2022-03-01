from torchvision import transforms as T

def transform(img):
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[.0, .0, .0], std=[255., 255., 255.]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return t(img)
