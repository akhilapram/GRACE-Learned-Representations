import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from Tilenet import make_tilenet
import pandas as pd
import pickle

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = make_tilenet()
        self.batch_size = 1
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss , return_none, l2,orig_loss = self.model.loss(batch[0],batch[1],batch[2])
        if return_none:
            print('returned None')
            return None
        self.log('train_loss',loss, on_step=True,on_epoch=True)
        self.log('train_l2',l2,on_step=True,on_epoch=True)
        self.log('train_main_loss',orig_loss,on_step=True,on_epoch=True)
        #print('logged training')
        return loss

    def validation_step(self, batch, batch_idx):
        loss , return_none,l2,orig_loss = self.model.loss(batch[0],batch[1],batch[2])
        if return_none:
            return None
        self.log('val_loss',loss,on_epoch=True,on_step=True)
        self.log('val_l2',l2,on_step=True,on_epoch=True)
        self.log('val_main_loss',orig_loss,on_step=True,on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss ,return_none, l2,orig_loss= self.model.loss(batch[0],batch[1],batch[2])
        if return_none:
            return None
        return loss
    
    def train_dataloader(self):
        dataset = CustomImageDataset(10000)
        return DataLoader(dataset,batch_size=500,num_workers=10)

    def val_dataloader(self):
        dataset = CustomImageDataset(1000)
        return DataLoader(dataset,batch_size=1000,num_workers=10)

    def test_dataloader():
        dataset = TestDataset() 
        return DataLoader(dataset,num_workers=0)

class TestDataset():
    def __init__(self):
        with open('Dataset.pkl', 'rb') as handle:
            self.dataset = pickle.load(handle)
        self.names=list(self.dataset)
        self.length = len(self.names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[self.names[idx]]).float().view(8,8,13).permute(2,0,1)


model = LitModel.load_from_checkpoint("../epoch=75-step=1519.ckpt").model
model = model.eval()
trainer = pl.Trainer()
dataset = TestDataset()

predictions=[]
with torch.no_grad():
    for idx in range(len(dataset)):
        pred=model(dataset[idx].unsqueeze(0))
        predictions.append(pred)

with open('../Model_Predictions.pkl', 'wb') as f:
    pickle.dump((dataset.names,predictions), f)
f.close()
