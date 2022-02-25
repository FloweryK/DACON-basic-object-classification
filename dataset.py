import os
import time
import cv2
from torch.utils.data import Dataset
from config import CLASSES


def load_paths(data_dir):
    paths = []

    print(f'dataset loading from {data_dir}')
    start = time.time()
    for class_name, target in CLASSES.items():
        class_dir = os.path.join(data_dir, class_name)

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            paths.append((file_path, target))
    end = time.time()
    print(f'dataset loaded ({(end-start):.2f}s)')
    
    return paths


class ObjectDataset(Dataset):
    def __init__(self, data_path):
        self.paths = load_paths(data_path)
        self.len = len(self.paths)
        self.cache = {}

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            # get path and target
            path, target = self.paths[index]

            # load image
            img = cv2.imread(path)

            # (optional) save result as cache
            self.cache[index] = (img, target)

            return img, target
        


if __name__ == "__main__":
    data_path = os.path.join("data", "train")
    object_dataset = ObjectDataset(data_path)

    # data loading test
    start = time.time()
    img, target = object_dataset[0]
    end = time.time()
    print('data:', img)
    print('data shape:', img.shape)
    print('target:', target)
    print('loading time:', (end - start)*1000, 'ms')

    # cache test
    _, _ = object_dataset[0]
    
