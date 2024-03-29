import os
import torch


class TrainerConfig:
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    lr = 1e-4
    weight_decay = 0.1
    batch_size = 64
    num_workers = 0
    num_epochs = 100
    shuffle=True
    pin_memory=True
    save_path='./model.pt'
    save_model=True

    print('TrainerConfig')
    print(f'device: {device}')
    print(f'lr: {lr}')
    print(f'weight_decay: {weight_decay}')
    print(f'batch_size: {batch_size}')
    print(f'num_workers: {num_workers}')
    print(f'num_epochs: {num_epochs}')
    print(f'shuffle: {shuffle}')
    print(f'pin_memory: {pin_memory}')
    print('\n')