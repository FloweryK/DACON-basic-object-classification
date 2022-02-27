import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class ObjectDataset(Dataset):
    def __init__(self, config):
        self.data = {}

        def load_data():
            # index for each data
            i = 0

            pbar = tqdm(enumerate(config.classes.items()), total=len(config.classes))
            for (_, (class_name, target)) in pbar:
                class_dir = os.path.join(config.data_dir, class_name)

                for file_name in os.listdir(class_dir):
                    # get file path
                    file_path = os.path.join(class_dir, file_name)

                    # read img and change from int to float
                    img = cv2.imread(file_path)
                    img = img.astype(np.float64)

                    # (optional) normalize
                    if config.is_norm:
                        img -= np.array(config.norm_mean)
                        img *= 1 / np.array(config.norm_std)
                    
                    # (optional) roll axis
                    img = np.moveaxis(img, -1, 0)

                    # save img to dataset
                    self.data[i] = (img, target)
                    i += 1
        
        print("loading data from:", config.data_dir)
        load_data()

        # save length to return the dataset size
        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    from config import DatasetConfig

    trainset = ObjectDataset(DatasetConfig())
    img, target = trainset[0]
    print(len(trainset))
    print(img.shape)
    print(target)