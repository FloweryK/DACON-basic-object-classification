import os
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T

def random_transform(img):
    t = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 30))
    ])
    return t(img)


class ObjectDataset(Dataset):
    def __init__(self, config, transform):
        self.data = {}
        self.config = config
        self.transform = transform
        
        print("loading data from:", config.data_dir)
        self.load_data()

        # save length to return the dataset size
        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.config.preload:
            return self.data[index]
        else:
            file_path, target = self.data[index]
            img = self.load_image(file_path)
            return (img, target)
    
    def load_image(self, file_path):
        img = Image.open(file_path)
        img = random_transform(img)
        img = self.transform(img)
        
        return img
    
    def load_data(self):
        # index for each data
        i = 0

        pbar = tqdm(enumerate(self.config.classes.items()), total=len(self.config.classes))
        for (_, (class_name, target)) in pbar:
            class_dir = os.path.join(self.config.data_dir, class_name)

            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)

                if self.config.preload:
                    # if preload, read img
                    img = self.load_image(file_path)

                    # save img to dataset
                    self.data[i] = (img, target)
                else :
                    # if not preload, only save path
                    self.data[i] = (file_path, target)

                i += 1


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from config import DatasetConfig

    # proper transform should be provided with each model
    def transform(img):
        t = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.0, .0, .0], std=[255., 255., 255.]),
        ])
        return t(img)

    dataset = ObjectDataset(DatasetConfig(), transform=transform)
    dataloader = DataLoader(dataset, batch_size=64)
    for imgs, target in dataloader:
        print(f'data sample: {imgs[0]}')
        print(f'data sample shape: {imgs[0].shape}')
        print(f'data sample target: {target}')
        print(f'dataset size: {len(dataset)}')
        print(f'dataloader size: {len(dataloader)}')