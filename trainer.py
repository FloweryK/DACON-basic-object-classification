import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(self, config, model, trainset, valiset, testset):
        self.config = config
        self.model = model
        self.trainset = trainset
        self.valiset = valiset
        self.testset = testset

        # TODO: locate this line to proper location
        self.model = self.model.to(self.config.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr)
        self.writer = SummaryWriter()
    
    def run(self):
        def run_epoch(epoch, dataset, mode):
            # flag for train mode or not
            is_train = mode == "train"
            n_correct = 0
            n_false = 0

            # set dataloader
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=self.config.shuffle,
                pin_memory=self.config.pin_memory,
            )

            # iterate through dataloader
            losses = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for it, (imgs, targets) in pbar:
                # put data into proper deivce
                imgs = imgs.to(self.config.device)
                targets = targets.to(self.config.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    prob = self.model(imgs)
                    loss = F.cross_entropy(prob.view(-1, prob.size(-1)), targets.view(-1))
                    losses.append(loss.item())

                    __check = torch.argmax(prob.view(-1, prob.size(-1)), axis=1) == targets.view(-1)
                    __correct = torch.sum(__check)
                    n_correct += __correct.item()
                    n_false += len(__check) - __correct.item()
                
                # if training, update parameters
                if is_train:
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # update desciption on progress bar
                loss_value = loss.item() if is_train else float(np.mean(losses))
                acc = n_correct / (n_correct + n_false)
                pbar.set_description(f'epoch {epoch} iter {it}: {mode} loss {loss_value:.5f} acc {acc:.5f}')
                
            # update tensorboard
            self.writer.add_scalar(f'Loss/{mode}', loss_value, epoch)
            self.writer.add_scalar(f'Acc/{mode}', acc, epoch)
            self.writer.add_graph(self.model, imgs)
        
        for epoch in range(self.config.num_epochs):
            run_epoch(epoch, self.trainset, "train")
            run_epoch(epoch, self.valiset, "vali")
            run_epoch(epoch, self.testset, "test")
                

if __name__ == "__main__":
    from torch.utils.data import random_split
    from models.CNNv1 import Model
    from dataset import ObjectDataset
    from config import DatasetConfig, TrainerConfig

    model = Model()
    dataset = ObjectDataset(DatasetConfig())
    n_train = int(len(dataset) * 0.8)
    n_vali = int(len(dataset) * 0.1)
    n_test = len(dataset) - (n_train + n_vali)

    trainset, valiset, testset = random_split(dataset, [n_train, n_vali, n_test])

    trainer = Trainer(TrainerConfig, model, trainset, valiset, testset)
    trainer.run()
    