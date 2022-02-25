import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ObjectDataset
from model import Model


class TrainerConfig:
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    batch_size = 64
    num_workers = 0
    num_epochs = 10

    print("=== configurations ===")
    print(f"device: {device}")
    print(f"lr: {lr}")
    print(f"batch_size: {batch_size}")
    print(f"num_worders: {num_workers}")


class Trainer:
    def __init__(self, model: nn.Module, trainset, testset, config):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.config = config

    def run(self, is_train):
        data = self.trainset if is_train else self.testset
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr)

        def run_epoch():
            loader = DataLoader(dataset=data,
                                shuffle=True,
                                pin_memory=True,
                                batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers)
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (imgs, targets) in pbar:
                # place data on the device
                imgs = imgs.to(self.config.device)
                targets = targets.to(self.config.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    prob = self.model(imgs)
                    loss = F.cross_entropy(prob.view(-1, prob.size(-1)), targets.view(-1))
                    losses.append(loss.item())
                
                if is_train:
                    self.model.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                pbar.set_description(f'epoch {epoch+1} iter {it}: train loss {loss.item():.5f}')
            
            
        for epoch in range(self.config.num_epochs):
            run_epoch()

                

if __name__ == "__main__":
    config = TrainerConfig()
    model = Model()
    trainset = ObjectDataset(os.path.join("data", "train"))
    trainer = Trainer(model, trainset, trainset, config)
    trainer.run(True)