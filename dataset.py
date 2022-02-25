import os
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
from config import CLASSES

class ObjectDataset(Dataset):
    def __init__(self, data_dir):
        self.data = {}

        def load_data():
            i = 0

            pbar = tqdm(enumerate(CLASSES.items()), total=len(CLASSES))
            for (_, (class_name, target)) in pbar:
                class_dir = os.path.join(data_dir, class_name)

                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    img = cv2.imread(file_path)

                    self.data[i] = (img, target)
                    i += 1
        
        print("loading data from:", data_dir)
        load_data()

        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    data_dir = os.path.join("data", "train")
    trainset = ObjectDataset(data_dir)
    img, target = trainset[0]
    print(len(trainset))
    print(img.shape)
    print(target)