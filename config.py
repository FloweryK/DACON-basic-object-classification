import torch

CLASSES = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}

class TrainerConfig:
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    batch_size = 64
    num_workers = 0
    num_epochs = 10