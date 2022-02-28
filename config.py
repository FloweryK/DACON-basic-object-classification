import os
import torch

class DatasetConfig:
    data_dir = os.path.join("data", "train")
    classes = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    aug_ratio = 2
    is_norm = True
    norm_mean = [0., 0., 0.]
    norm_std = [255., 255., 255.]
    # norm_mean = [113.86283869, 122.93301916, 125.26884795]
    # norm_std = [66.46033586, 61.90929531, 62.5340599]

class TrainerConfig:
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    lr = 1e-4
    weight_decay = 0.05
    batch_size = 64
    num_workers = 0
    num_epochs = 100
    shuffle=True
    pin_memory=True

    print(f'device: {device}')
    print(f'lr: {lr}')
    print(f'weight_decay: {weight_decay}')
    print(f'batch_size: {batch_size}')
    print(f'num_workers: {num_workers}')
    print(f'num_epochs: {num_epochs}')
    print(f'shuffle: {shuffle}')
    print(f'pin_memory: {pin_memory}')