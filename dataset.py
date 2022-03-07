import os
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ObjectDataset(Dataset):
    def __init__(self, paths, transform, augment=None):
        super().__init__()

        self.data = {}
        self.paths = paths
        self.augment = augment
        self.transform = transform

        self.load_image()
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]

    def load_image(self):
        pbar = tqdm(enumerate(self.paths), total=len(self.paths))
        for (i, (path, target)) in pbar:
            img = Image.open(path)

            if self.augment:
                for j in range(self.augment):
                    _img = random_transform(img)
                    _img = self.transform(img)
                    self.data[i*self.augment+j] = (_img, target)
            else:
                img = self.transform(img)
                self.data[i] = (img, target)


def random_transform(img):
    t = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(0, 30)),
    ])
    return t(img)


def load_datasets(transform, augment=4):
    # classes
    classes = {
        "airplane": 0, 
        "automobile": 1, 
        "bird": 2, 
        "cat": 3, 
        "deer": 4, 
        "dog": 5, 
        "frog": 6, 
        "horse": 7, 
        "ship": 8, 
        "truck": 9
    }
    
    # dataet path
    data_dir = os.path.join('data', 'train')

    # load file paths
    paths = []
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        for file_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file_name)
            target = classes[class_name]
            paths.append((img_path, target))

    # split into train, test, vali paths
    random.shuffle(paths)
    n_train = int(len(paths)*0.8)
    n_vali = int(len(paths)*0.1)
    paths_train = paths[:n_train]
    paths_vali = paths[n_train:n_train+n_vali]
    paths_test = paths[n_train+n_vali:]

    # create ObjectDatsaet object
    trainset = ObjectDataset(paths_train, transform=transform, augment=augment)
    valiset = ObjectDataset(paths_vali, transform=transform)
    testset = ObjectDataset(paths_test, transform=transform)
    return trainset, valiset, testset


if __name__ == "__main__":
    # proper transform should be provided with each model
    def transform(img):
        t = T.Compose([
            T.ToTensor(),
        ])
        return t(img)
    
    # load datasets
    trainset, valiset, testset = load_datasets(transform=transform, augment=3)
    
    print(len(trainset))
    print(len(valiset))
    print(len(testset))