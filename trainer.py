import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset import ObjectDataset
from model import Model


class TrainerConfig:
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    batch_size = 64
    num_workers = 0


class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config


    def train(self, is_train):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr)

        def run_epoch():
            self.model.train(is_train)
            loader = DataLoader(dataset=self.dataset, 
                                batch_size=self.config.batch_size, 
                                num_workers=self.config.num_workers,
                                shuffle=False, 
                                pin_memory=True,)
            
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, y) in pbar:
                pass
        
        for i in range(3):
            run_epoch()


if __name__ == "__main__":
    config = TrainerConfig()
    model = Model()
    trainset = ObjectDataset(os.path.join("data", "train"))
    trainer = Trainer(model, trainset, config)
    trainer.train(is_train=True)