import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T


class ObjectDataset(Dataset):
    def __init__(self, config, transform):
        self.data = {}

        def load_data():
            # index for each data
            i = 0

            pbar = tqdm(enumerate(config.classes.items()), total=len(config.classes))
            for (_, (class_name, target)) in pbar:
                class_dir = os.path.join(config.data_dir, class_name)

                for file_name in os.listdir(class_dir):
                    # read img
                    file_path = os.path.join(class_dir, file_name)
                    img = Image.open(file_path)

                    # transform
                    img = transform(img)

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